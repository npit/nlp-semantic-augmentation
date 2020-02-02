import pickle
from os.path import basename, exists, splitext

import numpy as np
from scipy import spatial

from semantic.semantic_resource import SemanticResource
from utils import debug, error, info, warning, write_pickled


class ContextEmbedding(SemanticResource):
    name = "context"

    def __init__(self, config):
        self.config = config
        error("Context embedding is TODO")
        # incompatible with embedding training
        error("Embedding context data missing: {}".format("Embedding train mode incompatible with semantic embeddings."),
              self.config.representation.name == "train")
        # read specific params
        self.embedding_aggregation = self.config.context_aggregation
        self.representation_dim = self.config.representation.dimension
        self.context_threshold = self.config.context_threshold
        self.context_file = self.config.context_file
        # calculate the synset embeddings path
        SemanticResource.__init__(self)
        if not any([x for x in self.load_flags]):
            error("Failed to load semantic embeddings context.")

    def set_name(self):
        SemanticResource.set_name(self)
        thr = ""
        if self.context_threshold:
            thr += "_thresh{}".format(self.context_threshold)
        self.name += "_ctx{}_emb{}{}".format(basename(splitext(self.context_file)[0]), self.config.representation.name, thr)

    def get_raw_path(self):
        return self.context_file

    def handle_raw(self, raw_data):
        self.semantic_context = {}
        # apply word frequency thresholding, if applicable
        if self.context_threshold is not None:
            num_original = len(raw_data.items())
            info("Limiting the {} reference context concepts with a word frequency threshold of {}".format(num_original, self.context_threshold))
            self.semantic_context = {s: wl for (s, wl) in raw_data.items() if len(wl) >= self.context_threshold}
            info("Ended up with context information for {} concepts.".format(len(self.semantic_context)))
        else:
            self.semantic_context = raw_data
        # set the loaded concepts as the reference concept list
        self.reference_concepts = list(sorted(self.semantic_context.keys()))
        if not self.reference_concepts:
            error("Frequency threshold of synset context: {} resulted in zero reference concepts".format(self.context_threshold))
        info("Applied {} reference concepts from pre-extracted synset words".format(len(self.reference_concepts)))
        return self.semantic_context

        # serialize
        write_pickled(self.sem_embeddings_path, raw_data)

    def fetch_raw(self, path):
        # load the concept-words list
        error("Embedding context data missing: {}".format(self.context_file), not exists(self.context_file))
        with open(self.context_file, "rb") as f:
            data = pickle.load(f)
        return data

    def map_text(self, embedding, dataset):
        self.embedding = embedding
        self.compute_semantic_embeddings()
        # kdtree for fast lookup
        self.kdtree = spatial.KDTree(self.concept_embeddings)
        SemanticResource.map_text(self, embedding, dataset)

    def lookup(self, candidate):
        word, _ = candidate
        if self.do_cache and word in self.word_concept_embedding_cache:
            concept = self.word_concept_embedding_cache[word]
        else:
            word_embedding = self.embedding.get_embeddings([word])
            _, conc_idx = self.kdtree.query(word_embedding)
            if conc_idx is None or len(conc_idx) == 0:
                return {}
            concept = self.reference_concepts[int(conc_idx)]
        # no spreading activation defined here.
        return {concept: 1}

    def handle_raw_serialized(self, raw_serialized):
        self.loaded_raw_serialized = True
        self.concept_embeddings, self.reference_concepts = raw_serialized
        debug("Read concept embeddings shape: {}".format(self.concept_embeddings.shape))

    # generate semantic embeddings from words associated with an concept
    def compute_semantic_embeddings(self):
        if self.loaded_raw_serialized:
            return
        info("Computing semantic embeddings, using {} embeddings of dim {}.".format(self.embedding.name, self.representation_dim))
        retained_reference_concepts = []
        self.concept_embeddings = np.ndarray((0, self.representation_dim), np.float32)
        for s, concept in enumerate(self.reference_concepts):
            # get the embeddings for the words in the concept's context
            words = self.semantic_context[concept]
            debug("Reference concept {}/{}: {}, context words: {}".format(s + 1, len(self.reference_concepts), concept, len(words)))
            word_embeddings = self.embedding.get_embeddings(words)
            if len(word_embeddings) == 0:
                continue
            # aggregate
            if self.embedding_aggregation == "avg":
                embedding = np.mean(word_embeddings.as_matrix(), axis=0)
                self.concept_embeddings = np.vstack([self.concept_embeddings, embedding])
            else:
                error("Undefined semantic embedding aggregation:{}".format(self.embedding_aggregation))

            retained_reference_concepts.append(concept)

        num_dropped = len(self.reference_concepts) - len(retained_reference_concepts)
        if num_dropped > 0:
            info("Discarded {} / {} concepts resulting in {}, due to no context words existing in read embeddings.".format(num_dropped, len(self.reference_concepts), len(retained_reference_concepts)))
        self.reference_concepts = retained_reference_concepts
        # save results
        info("Writing semantic embeddings to {}".format(self.serialization_path))
        write_pickled(self.serialization_path, [self.concept_embeddings, self.reference_concepts])
        self.loaded_raw_serialized = True

    def get_semantic_embeddings(self):
        semantic_document_vectors = np.ndarray((0, self.representation_dim), np.float32)
        # get raw semantic frequencies
        for d in range(len(self.concept_freqs)):
            for doc_index, doc_dict in enumerate(self.concept_freqs[d]):
                doc_sem_embeddings = np.ndarray((0, self.semantic_representation_dim), np.float32)
                if not doc_dict:
                    warning("Attempting to get semantic embedding vectors of document {}/{} with no semantic mappings. Defaulting to zero vector.".format(doc_index + 1, len(self.concept_freqs[d])))
                    doc_vector = np.zeros((self.semantic_representation_dim,), np.float32)
                else:
                    # gather semantic embeddings of all document concepts
                    for concept in doc_dict:
                        concept_index = self.concept_order.index(concept)
                        doc_sem_embeddings = np.vstack([doc_sem_embeddings, self.concept_embeddings[concept_index, :]])
                    # aggregate them
                    if self.semantic_embedding_aggregation == "avg":
                        doc_vector = np.mean(doc_sem_embeddings, axis=0)
                semantic_document_vectors[d].append(doc_vector)

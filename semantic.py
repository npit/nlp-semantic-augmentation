from os.path import join, exists, splitext, basename
from os import listdir
from dataset import Dataset
import pickle
import nltk
from utils import tictoc, error, info, debug, warning, write_pickled, read_pickled, shapes_list
import numpy as np
from serializable import Serializable
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import json
import urllib
from scipy import spatial

import defs


class SemanticResource(Serializable):
    dir_name = "semantic"
    semantic_name = None
    name = None
    do_spread_activation = False
    loaded_vectorized = False

    word_concept_lookup_cache = {}
    word_concept_embedding_cache = {}

    concept_freqs = []
    concept_tfidf_freqs = []
    dataset_freqs = []
    dataset_minmax_freqs = []
    assignments = {}
    concept_context_word_threshold = None

    reference_concepts = None
    do_limit = False

    disambiguation = None
    pos_tag_mapping = {}

    def get_appropriate_config_names(self):
        semantic_names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.semantic.limit.none]
        if self.do_limit:
            filter_vals.append(defs.semantic.limit.to_string(self.config))
        # any combo of weights, since they're all stored
        weight_vals = defs.semantic.weights.avail()
        for w in weight_vals:
            for f in filter_vals:
                semantic_names.append(
                    SemanticResource.get_semantic_name(self.config, filtering=f, sem_weights=w))
                debug("Semantic config candidate: {}".format(semantic_names[-1]))

        return semantic_names



    def __init__(self, config):
        Serializable.__init__(self, self.dir_name)
        self.set_parameters()
        config_names = self.get_appropriate_config_names()
        for s, semantic_name in enumerate(config_names):
            debug("Attempting to load semantic info from source {}/{}: {}".format(s+1, len(config_names), semantic_name))
            self.semantic_name = semantic_name
            self.form_name()
            self.set_serialization_params()
            # add extras
            self.serialization_path_vectorized = self.serialization_path_preprocessed + ".vectorized"
            self.data_paths.insert(0, self.serialization_path_vectorized)
            self.read_functions.insert(0, read_pickled)
            self.handler_functions.insert(0, self.handle_vectorized)
            self.acquire2(fatal_error=False)
            if any(self.load_flags):
                info("Loaded by using semantic name: {}".format(semantic_name))
                break
        # restore correct config
        self.semantic_name = SemanticResource.get_semantic_name(self.config)
        info("Restored specifid semantic name to : {}".format(self.semantic_name))
        self.form_name()


    def create(config):
        name = config.semantic.name
        if name == Wordnet.name:
            return Wordnet(config)
        if name == GoogleKnowledgeGraph.name:
            return GoogleKnowledgeGraph(config)
        if name == ContextEmbedding.name:
            return ContextEmbedding(config)
        if name == Framenet.name:
            return Framenet(config)
        error("Undefined semantic resource: {}".format(name))
    pass

    def get_vectors(self):
        return self.semantic_document_vectors


    def generate_vectors(self):
        if self.loaded_vectorized:
            info("Skipping generating, since loaded vectorized data already.")
            return
        if self.embedding.loaded_enriched():
            info("Skipping generating, since loaded enriched data already.")
            return
        # map dicts to vectors
        if not self.dataset_freqs:
            error("Attempted to generate semantic vectors from empty containers")
        info("Generating semantic vectors.".format(self.semantic_weights))
        concept_order = sorted(self.reference_concepts)
        self.dimension = len(concept_order)
        semantic_document_vectors = [np.ndarray((0, self.dimension), np.float32) for _ in range(len(self.concept_freqs))]
        if self.semantic_weights == "frequencies":
            # get raw semantic frequencies
            for d in range(len(self.concept_freqs)):
                for doc_dict in self.concept_freqs[d]:
                    doc_vector = np.asarray([[doc_dict[s] if s in doc_dict else 0 for s in concept_order]], np.float32)
                    semantic_document_vectors[d] = np.append(semantic_document_vectors[d], doc_vector, axis=0)
        elif self.semantic_weights == "tfidf":
            # get tfidf weights
            for d in range(len(self.concept_tfidf_freqs)):
                for doc_dict in self.concept_tfidf_freqs[d]:
                    doc_vector = np.asarray([[doc_dict[s] if s in doc_dict else 0 for s in concept_order]], np.float32)
                    semantic_document_vectors[d] = np.append(semantic_document_vectors[d], doc_vector, axis=0)

        elif self.semantic_weights == "embeddings":
            error("Embedding information requires the context_embedding semantic disambiguation. It is {} instead.".format(
                        self.disambiguation), condition=self.disambiguation!="context_embedding")
            semantic_document_vectors = self.get_semantic_embeddings()
        else:
            error("Unimplemented semantic vector method: {}.".format(self.semantic_weights))

        self.semantic_document_vectors = semantic_document_vectors
        write_pickled(self.serialization_path_vectorized, [self.semantic_document_vectors, concept_order])




    # function to get a concept from a word, using the wordnet api
    # and a local word cache. Updates concept frequencies as well.
    def get_concept(self, word_information, freqs, force_reference_concepts = False):
        word, _ = word_information
        if word_information in self.word_concept_lookup_cache:
            concept_activations = self.word_concept_lookup_cache[word_information]
            concept_activations = self.restrict_to_reference(force_reference_concepts, concept_activations)
            freqs = self.update_frequencies(concept_activations, freqs)
        else:
            concept_activations = self.lookup(word_information)
            if not concept_activations:
                return None, freqs
            concept_activations = self.restrict_to_reference(force_reference_concepts, concept_activations)
            freqs = self.update_frequencies(concept_activations, freqs)
            # populate cache
            self.word_concept_lookup_cache[word_information] = concept_activations

        if word not in self.assignments:
            self.assignments[word] = concept_activations
        return concept_activations, freqs

    def get_semantic_name(config, filtering=None, sem_weights=None):
        if not config.has_semantic():
            return None
        if filtering is None:
            filtering = defs.semantic.limit.to_string(config)
        if sem_weights is None:
            sem_weights = defs.semantic.weights.to_string(config)
        disambig = "disam{}".format(config.semantic.disambiguation)
        semantic_name = "{}_{}_{}_{}".format(config.semantic.name, sem_weights,filtering, disambig)
        if config.semantic.spreading_activation:
            steps, decay = config.semantic.spreading_activation
            semantic_name += "_spread{}-{}".format(steps, decay)
        return semantic_name

    def handle_vectorized(self, data):
        self.semantic_document_vectors, self.concept_order = data
        self.loaded_preprocessed = True
        self.loaded_vectorized = True
        debug("Read vectorized concept docs shapes: {}, {} and concept order: {}".format(*shapes_list(self.semantic_document_vectors), len(self.concept_order)))

    def get_all_preprocessed(self):
        return [self.assignments, self.concept_freqs, self.dataset_freqs, self.concept_tfidf_freqs, self.reference_concepts]

    def get_raw_path(self):
        return None

    def restrict_to_reference(self, do_force, activations):
        if do_force:
            activations = {s: activations[s] for s in activations if s in self.reference_concepts}
        return activations

    def lookup(self, candidate):
        error("Attempted to lookup from the base class")


    def set_parameters(self):
        if self.config.semantic.limit is not None:
            self.do_limit = True
            self.limit_type, self.limit_number = self.config.semantic.limit

        self.semantic_weights = self.config.semantic.weights
        self.semantic_unit = self.config.semantic.unit
        self.disambiguation = self.config.semantic.disambiguation.lower()
        if self.config.semantic.spreading_activation:
            self.do_spread_activation = True
            self.spread_steps, self.spread_decay = self.config.semantic.spreading_activation[0], \
                                                   self.config.semantic.spreading_activation[1]

        self.dataset_name = Dataset.get_limited_name(self.config)
        self.semantic_name = SemanticResource.get_semantic_name(self.config)
        self.form_name()


    # make name string from components
    def form_name(self):
        self.name = "{}_{}".format(self.dataset_name, self.semantic_name)

    def fetch_raw(self, dummy_input):
        return None

    def assign_embedding(self, embedding):
        self.embedding = embedding


    # prune semantic information units wrt a frequency threshold
    def apply_limiting(self, freq_dict_list, dataset_freqs, force_reference=False):
        info("Applying concept filtering with a limiting config of {}-{}".format(self.limit_type, self.limit_number))
        if self.limit_type == "frequency":
            # discard concepts with a lower frequency than the specified threshold
            with tictoc("Dataset-level frequency filtering"):
                # delete from dataset-level dicts
                concepts_to_delete = set()
                for concept in dataset_freqs:
                    if dataset_freqs[concept] < self.limit_number:
                        concepts_to_delete.add(concept)
            return self.check_delete_concepts(freq_dict_list, dataset_freqs, force_reference, concepts_to_delete)

        elif self.limit_type == "first":
            # keep only the specified number of concepts with the highest weight
            sorted_dset_freqs = sorted(dataset_freqs, reverse=True, key = lambda x : dataset_freqs[x])
            concepts_to_delete = sorted_dset_freqs[self.limit_number:]
            return self.check_delete_concepts(freq_dict_list, dataset_freqs, force_reference, concepts_to_delete)
        else:
            error("Undefined semantic limiting type: {}".format(self.limit_type))


    def check_delete_concepts(self, freq_dict_list, dataset_freqs, force_reference, concepts_to_delete):
        # if forcing reference, we can override the freq threshold for these concepts
        if force_reference:
            orig_num = len(concepts_to_delete)
            concepts_to_delete = [x for x in concepts_to_delete if x not in self.reference_concepts]
            info("Limiting concepts-to-delete from {} to {} due to forcing to reference concept set".format(orig_num, len(concepts_to_delete)))
        if not concepts_to_delete:
            return  freq_dict_list, dataset_freqs
        info("Will remove {}/{} concepts due to a limiting config of {}-{}".format(
            len(concepts_to_delete), len(dataset_freqs), self.limit_type, self.limit_number))
        with tictoc("Document-level frequency filtering"):
            # delete
            for concept in concepts_to_delete:
                del dataset_freqs[concept]
                for doc_dict in freq_dict_list:
                    if concept in doc_dict:
                        del doc_dict[concept]
        info("Concept filtering resulted in {} concepts.".format(len(dataset_freqs)))
        return  freq_dict_list, dataset_freqs


    # merge list of document-wise frequency dicts
    # to a single, dataset-wise frequency dict
    def doc_to_dset_freqs(self, freq_dict_list, force_reference = False):
        dataset_freqs = {}
        for doc_dict in freq_dict_list:
            for concept, freq in doc_dict.items():
                if concept not in dataset_freqs:
                    dataset_freqs[concept] = 0
                dataset_freqs[concept] += freq
        # frequency filtering, if defined
        if self.do_limit:
            freq_dict_list, dataset_freqs = self.apply_limiting(freq_dict_list, dataset_freqs, force_reference)

        # complete document-level freqs with zeros for dataset-level concepts missing in the document level
        #for d, doc_dict in enumerate(freq_dict_list):
        #    for concept in [s for s in dataset_freqs if s not in doc_dict]:
        #        freq_dict_list[d][concept] = 0
        return dataset_freqs, freq_dict_list


    # tf-idf computation
    def compute_tfidf_weights(self, current_concept_freqs, dataset_freqs, force_reference=False):
        # compute tf-idf
        with tictoc("tf-idf computation"):
            tfidf_freqs = []
            for doc_dict in range(len(current_concept_freqs)):
                ddict = {}
                for synset in current_concept_freqs[doc_dict]:
                    if dataset_freqs[synset] > 0:
                        ddict[synset] = current_concept_freqs[doc_dict][synset] / dataset_freqs[synset]
                    else:
                        ddict[synset] = 0

                tfidf_freqs.append(ddict)
            self.concept_tfidf_freqs.append(tfidf_freqs)

    # map a single dataset portion
    def map_dset(self, dset_words_pos, store_reference_concepts = False, force_reference_concepts = False):
        if force_reference_concepts:
            # restrict discoverable concepts to a predefined selection
            # used for the test set where not encountered concepts are unusable
            info("Restricting concepts to the reference synset set of {} entries.".format(len(self.reference_concepts)))

        current_synset_freqs = []
        with tictoc("Document-level mapping and frequency computation"):
            for wl, word_info_list in enumerate(dset_words_pos):
                debug("Semantic processing for document {}/{}, which has {} words".format(wl+1, len(dset_words_pos), len(word_info_list)))
                doc_freqs = {}
                for w, word_info in enumerate(word_info_list):
                    synset_activations, doc_freqs = self.get_concept(word_info, doc_freqs, force_reference_concepts)
                    if not synset_activations: continue
                if not doc_freqs:
                    warning("No semantic information extracted for document {}/{}".format(wl+1, len(dset_words_pos)))
                current_synset_freqs.append(doc_freqs)

        # merge to dataset-wise synset frequencies
        with tictoc("Dataset-level frequency computation"):
            dataset_freqs, current_synset_freqs = self.doc_to_dset_freqs(current_synset_freqs, force_reference = force_reference_concepts)
            self.dataset_freqs.append(dataset_freqs)
            self.concept_freqs.append(current_synset_freqs)


        if store_reference_concepts:
            self.reference_concepts = set((dataset_freqs.keys()))

        # compute tfidf
        if self.semantic_weights == "tfidf":
            self.compute_tfidf_weights(current_synset_freqs, dataset_freqs, force_reference=force_reference_concepts)


    # apply disambiguation to choose a single semantic unit from a collection of such
    def disambiguate(self, concepts, word_information, override=None):
        disam = self.disambiguation if not override else override
        if disam == "first":
            return [concepts[0]]
        elif disam == 'pos':
            # take part-of-speech tags into account
            word, word_pos = word_information
            word_pos = word_pos[:2]
            # if not exist, revert to first
            if word_pos is None:
                return self.disambiguate(concepts, word_information, override="first")
            if word_pos not in self.pos_tag_mapping:
                return self.disambiguate(concepts, word_information, override="first")
            # if encountered matching pos, get it.
            for concept in concepts:
                if concept._pos == self.pos_tag_mapping[word_pos]:
                    return [concept]
            # no pos match, revert to first
            return self.disambiguate(concepts, word_information, override="first")
        else:
            error("Undefined disambiguation method: " + self.disambiguation)


    # function to map words to wordnet concepts
    def map_text(self, embedding, dataset):
        self.embedding = embedding
        if self.loaded_preprocessed:
            info("Skipping mapping text due to preprocessed data already loaded.")
            return
        if self.embedding.loaded_enriched():
            info("Skipping mapping text due to enriched data already loaded.")
            return

        # process the data
        dataset_pos = dataset.get_pos(self.embedding.get_present_word_indexes())
        self.concept_freqs = []
        for d, dset in enumerate(dataset_pos):
            info("Extracting {} semantic information from dataset {}/{}".format(self.name, d+1, len(dataset_pos)))
            # process data within a dataset portion
            # should store reference concept in the train portion (d==0), but only if a reference has not
            # been already defined, e.g. via semantic embedding precomputations
            store_as_reference = (d == 0 and not self.reference_concepts)
            # should force mapping to the reference if a reference has been defined
            force_reference = bool(self.reference_concepts)
            self.map_dset(dset, store_as_reference, force_reference)

        # write results: word assignments, raw, dataset-wise and tf-idf weights
        info("Writing semantic assignment results to {}.".format(self.serialization_path))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())
        info("Semantic mapping completed.")

    def update_frequencies(self, activations, frequencies):
        for s, act in activations.items():
            if s not in frequencies:
                frequencies[s] = 0
            frequencies[s] += act
        return frequencies



    def spread_activation(self, synset, steps_to_go, current_decay):
        error("Attempted to call abstract spread activation for semantic resource {}.".format(self.name))

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.assignments, self.concept_freqs, self.dataset_freqs, self.concept_tfidf_freqs, self.reference_concepts = preprocessed
        debug("Read preprocessed concept docs shapes: {}, {}".format(*list(map(len,self.concept_freqs))))



class Wordnet(SemanticResource):
    name = "wordnet"

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        SemanticResource.__init__(self, config)

        # map nltk pos maps into meaningful wordnet ones
        self.pos_tag_mapping = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADJ, "RB": wn.ADV}



    def fetch_raw(self, dummy_input):
        if not self.base_name in listdir(nltk.data.find("corpora")):
            nltk.download("wordnet")
        return None


    def handle_raw_serialized(self, raw_serialized):
        pass

    def handle_raw(self, raw_data):
        pass


    def lookup(self, word_information):
        word, _  = word_information
        synsets = wn.synsets(word)
        if not synsets:
            return {}
        synsets = self.disambiguate(synsets, word_information)
        activations = {synset._name: 1 for synset in synsets}
        if self.do_spread_activation:
            # climb the hypernym ladder
            hyper_activations = self.spread_activation(synsets, self.spread_steps, 1)
            activations = {**activations, **hyper_activations}
        return activations

    def spread_activation(self, synsets, steps_to_go, current_decay):
        if steps_to_go == 0:
            return
        activations = {}
        # current weight value
        new_decay = current_decay * self.spread_decay
        for synset in synsets:
            # get hypernyms of synset
            for hyper in synset.hypernyms():
                activations[hyper._name] = current_decay
                hypers = self.spread_activation([hyper], steps_to_go-1, new_decay)
                if hypers:
                    activations = {**activations, **hypers}
        return activations


class ContextEmbedding(SemanticResource):
    name = "context"

    def __init__(self, config):
        self.config = config
        # incompatible with embedding training
        error("Embedding context data missing: {}".format("Embedding train mode incompatible with semantic embeddings."),
              self.config.embedding.name == "train")
        # read specific params
        self.embedding_aggregation = self.config.semantic.context_aggregation
        self.embedding_dim = self.config.embedding.dimension
        self.context_threshold = self.config.semantic.context_threshold
        self.context_file = self.config.semantic.context_file
        # calculate the synset embeddings path
        SemanticResource.__init__(self, config)
        if not any([x for x in self.load_flags]):
            error("Failed to load semantic embeddings context.")


    def form_name(self):
        SemanticResource.form_name(self)
        thr=""
        if self.context_threshold:
            thr += "_thresh{}".format(self.context_threshold)
        self.name += "_ctx{}_emb{}{}".format(basename(splitext(self.context_file)[0]), self.config.embedding.name, thr)


    def get_raw_path(self):
        return self.context_file

    def handle_raw(self, raw_data):
        self.semantic_context = {}
        # apply word frequency thresholding, if applicable
        if self.context_threshold is not None:
            num_original = len(raw_data.items())
            info("Limiting the {} reference context concepts with a word frequency threshold of {}".format(num_original, self.context_threshold))
            self.semantic_context = {s: wl for (s,wl) in raw_data.items() if len(wl) >= self.context_threshold}
            info("Ended up with context information for {} concepts.".format(len(self.semantic_context)))
        else:
            self.semantic_context = raw_data;
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
        if word in self.word_concept_embedding_cache:
            concept = self.word_concept_embedding_cache[word]
        else:
            word_embedding = self.embedding.get_embeddings([word])
            _, conc_idx = self.kdtree.query(word_embedding)
            if conc_idx is None or len(conc_idx) == 0:
                return {}
            concept = self.reference_concepts[int(conc_idx)]
        # no spreading activation defined here.
        return { concept: 1}

    def handle_raw_serialized(self, raw_serialized):
        self.loaded_raw_serialized = True
        self.concept_embeddings, self.reference_concepts = raw_serialized
        debug("Read concept embeddings shape: {}".format(self.concept_embeddings.shape))


    # generate semantic embeddings from words associated with an concept
    def compute_semantic_embeddings(self):
        if self.loaded_raw_serialized:
            return
        info("Computing semantic embeddings, using {} embeddings of dim {}.".format(self.embedding.name, self.embedding_dim))
        retained_reference_concepts = []
        self.concept_embeddings = np.ndarray((0, self.embedding_dim), np.float32)
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
        semantic_document_vectors = np.ndarray((0, self.embedding_dim), np.float32)
        # get raw semantic frequencies
        for d in range(len(self.concept_freqs)):
            for doc_index, doc_dict in enumerate(self.concept_freqs[d]):
                doc_sem_embeddings = np.ndarray((0, self.semantic_embedding_dim), np.float32)
                if not doc_dict:
                    warning("Attempting to get semantic embedding vectors of document {}/{} with no semantic mappings. Defaulting to zero vector.".format(doc_index + 1, len(self.concept_freqs[d])))
                    doc_vector = np.zeros((self.semantic_embedding_dim,), np.float32)
                else:
                    # gather semantic embeddings of all document concepts
                    for concept in doc_dict:
                        concept_index = self.concept_order.index(concept)
                        doc_sem_embeddings = np.vstack([doc_sem_embeddings, self.concept_embeddings[concept_index, :]])
                    # aggregate them
                    if self.semantic_embedding_aggregation == "avg":
                        doc_vector = np.mean(doc_sem_embeddings, axis=0)
                semantic_document_vectors[d].append(doc_vector)


class GoogleKnowledgeGraph(SemanticResource):
    name = "googlekt"

    query_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    key = None

    def __init__(self, config):
        self.config = config
        self.key = config.misc.keys["googleapi"]
        self.query_params = {
            'limit': 10,
            'indent': True,
            'key': self.key,
        }
        SemanticResource.__init__(self, config)

    def lookup(self, candidate):
        word, pos_info = candidate
        self.query_params["query"] = word
        url = self.query_url + '?' + urllib.parse.urlencode(self.query_params)
        response = json.loads(urllib.request.urlopen(url).read())
        names, hypers, scores = [], [], []
        for element in response['itemListElement']:
            results = element['result']
            if "name" not in results:
                continue
            scores.append(element['resultScore'])
            names.append(results['name'])
            hypers.append(results['@type'])
            # descr = results['description']
            # detailed_descr = results['detailedDescription'] if 'detailedDescription' in results else None

        names = self.disambiguate(names, candidate)
        activations = {n: 1 for n in names}

        if self.do_spread_activation:
            for name in names:
                idx = names.index(name)
                hyps = hypers[idx]
                activations[name] = 1
                if self.do_spread_activation:
                    current_decay = self.spread_decay
                    for h, hyp in enumerate(hyps):
                        if h+1 > self.spread_steps:
                            break
                        activations[hyp] = current_decay
                        current_decay *= self.spread_decay

        return activations


class Framenet(SemanticResource):
    name = "framenet"
    relations_to_spread = ["Inheritance"]

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        SemanticResource.__init__(self, config)
        # map nltk pos maps into meaningful framenet ones
        self.pos_tag_mapping = {"VB": "V", "NN": "N", "JJ": "A", "RB": "ADV"}

    def fetch_raw(self, dummy_input):
        if not self.base_name + "_v17" in listdir(nltk.data.find("corpora")):
            nltk.download("framenet_v17")
        return None

    def lookup(self, candidate):
        # http://www.nltk.org/howto/framenet.html
        word, word_pos = candidate
        # in framenet, pos-disambiguation is done via the lookup
        if self.disambiguation == defs.semantic.disam.pos:
            frames = self.lookup_with_POS(candidate)
        else:
            frames = fn.frames_by_lemma(word)
            if not frames:
                return None
            frames = self.disambiguate(frames, candidate)
        if not frames:
            return None
        activations = {x.name: 1 for x in frames}
        if self.do_spread_activation:
            parent_activations = self.spread_activation(frames, self.spread_steps, 1)
            activations = {**activations, **parent_activations}
        return activations

    def lookup_with_POS(self, candidate):
        word, word_pos = candidate
        if word_pos in self.pos_tag_mapping:
            word += "." + self.pos_tag_mapping[word_pos]
        frames = fn.frames_by_lemma(word)
        if not frames:
            return None
        return self.disambiguate(frames, candidate, override=defs.semantic.disam.first)


    def get_related_frames(self, frame):
        # get just parents
        return [fr.Parent for fr in frame.frameRelations if fr.type.name == "Inheritance" and fr.Child == frame]

    def spread_activation(self, frames, steps_to_go, current_decay):
        if steps_to_go == 0:
            return
        activations = {}
        current_decay *= self.spread_decay
        for frame in frames:
            related_frames = self.get_related_frames(frame)
            for rel in related_frames:
                activations[rel.name] = current_decay
                parents = self.spread_activation([rel], steps_to_go-1, current_decay)
                if parents:
                    activations = {**activations, **parents}
        return activations



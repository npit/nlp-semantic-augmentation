import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import defs
from representation.embedding import Embedding
from representation.representation import Representation
from utils import info, tictoc, write_pickled


class DocumentEmbedding(Embedding):
    """Document embedding class based on the doc2vec algorithm implemented by GenSim"""
    name = "doc2vec"

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.representation.name
        Embedding.__init__(self)

    def set_resources(self):
        if self.loaded():
            return
        # load existing word-level embeddings -- only when computing from scratch
        csv_mapping_name = "{}/{}.wordembeddings.csv".format(self.raw_data_dir, self.base_name)
        self.resource_paths.append(csv_mapping_name)
        self.resource_read_functions.append(self.read_raw_embedding_mapping)
        self.resource_handler_functions.append(lambda x: x)

    # nothing to load, can be computed on the fly
    def fetch_raw(self, path):
        pass

    def fit_doc2vec(self, train_word_lists, multi_labels):
        # learn document vectors from the training dataset
        tagged_docs = [TaggedDocument(doc, i) for doc, i in zip(train_word_lists, multi_labels)]
        model = Doc2Vec(tagged_docs, vector_size=self.dimension, min_count=0, window=5, dm=1)
        # update_term_frequency word vectors with loaded elements
        info("Updating word embeddings with loaded vectors")
        words = [w for w in self.embeddings.index.to_list() if w in model.wv.vocab.keys()]
        for word in words:
            word_index = model.wv.index2entity.index(word)
            model.wv.syn0[word_index] = self.embeddings.loc[word]
        # model.wv.add(words, self.embeddings.loc[words], replace=True)
        model.train(tagged_docs, epochs=10, total_examples=len(tagged_docs))
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        del self.embeddings
        return model

    def map_text(self):
        if self.loaded_preprocessed or self.loaded_aggregated:
            return
        info("Mapping to {} embeddings.".format(self.name))
        word_lists = self.text
        labels = self.labels[0]
        if type(labels[0]) is not list:
            labels = [[l] for l in labels]
        train_words = [wp[0] for doc in word_lists[0] for wp in doc]
        d2v = self.fit_doc2vec(train_words, labels)

        text_bundles = self.text
        self.dataset_vectors = [[], []]

        # loop over input text bundles (e.g. train & test)
        for dset_idx in range(len(text_bundles)):
            dset_word_list = word_lists[dset_idx]
            with tictoc("Embedding mapping for text bundle {}/{}".format(dset_idx + 1, len(text_bundles))):
                info("Mapping text bundle {}/{}: {} texts".format(dset_idx + 1, len(text_bundles), len(text_bundles[dset_idx])))
                # num_documents = len(text_bundles[dset_idx])
                for doc_word_pos in dset_word_list:
                    doc_words = [wp[0] for wp in doc_word_pos]
                    # debug("Inferring word list:{}".format(doc_words))
                    self.dataset_vectors[dset_idx].append(d2v.infer_vector(doc_words))
            self.dataset_vectors[dset_idx] = np.array(self.dataset_vectors[dset_idx])

        self.set_constant_elements_per_instance()

        # assign
        num_train, num_test = len(self.dataset_vectors[0]), len(self.dataset_vectors[1])
        indexes = (list(range(num_train)), list(range(num_train, num_train + num_test)))
        self.embeddings = np.vstack(self.dataset_vectors)
        self.dataset_vectors = indexes
        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def aggregate_instance_vectors(self):
        pass

    def set_params(self):
        Embedding.set_params(self)
        # define compatible aggregations
        self.compatible_aggregations = [defs.alias.none, None]
        self.compatible_sequence_lengths = [defs.sequence_length.unit]

    def process_component_inputs(self):
        Representation.process_component_inputs(self)
        if self.loaded_aggregated or self.loaded_preprocessed:
            return
        self.labels = self.inputs.get_labels().instances

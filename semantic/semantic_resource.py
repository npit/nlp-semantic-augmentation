from os import makedirs
from os.path import dirname, exists, join

import defs
import numpy as np
from bundle.bundle import DataPool
from bundle.datatypes import Text, Numeric
from bundle.datausages import *
from component.component import Component
from defs import is_none
from representation.bag import Bag
from serializable import Serializable
from utils import (debug, error, info, read_pickled, shapes_list, tictoc,
                   warning, write_pickled)


class SemanticResource(Serializable):
    dir_name = "semantic"
    component_name = "semantic"
    semantic_name = None
    name = None
    do_spread_activation = False
    loaded_vectorized = False

    lookup_cache = {}
    hypernym_cache = {}
    word_concept_embedding_cache = {}

    concept_freqs = []
    concept_context_word_threshold = None

    reference_concepts = None

    semantic_vector_indices = None
    semantic_epi = None

    disambiguation = None
    pos_tag_mapping = {}
    representation = None

    do_cache = True

    data_names = ["concept_weights", "global_weights", "reference_concepts"]
    data_names_vectorized = ["semantic_vectors", "semantic_vector_indices", "elements_per_instance"]

    consumes=Text.name
    produces=Numeric.name

    @staticmethod
    def get_available():
        return [cls.name for cls in SemanticResource.__subclasses__()]

    def set_additional_serialization_sources(self):
        self.serialization_path_vectorized = self.serialization_path_preprocessed + ".vectorized"
        self.data_paths.insert(0, self.serialization_path_vectorized)
        self.read_functions.insert(0, read_pickled)
        self.handler_functions.insert(0, self.handle_vectorized)

    def set_multiple_config_names(self):
        return
        semantic_names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.limit.none]
        if not is_none(self.config.limit):
            filter_vals.append(self.config.limit)
        # any combo of weights, since local and global freqs are stored
        weight_vals = defs.weights.avail

        for w in weight_vals:
            for f in filter_vals:
                conf = self.config.get_copy()
                conf.weights = w
                conf.limit = f
                candidate_name = self.generate_name(conf, self.source_name)
                debug("Semantic config candidate: {}".format(candidate_name))
                semantic_names.append(candidate_name)
        self.multiple_config_names = semantic_names

    def __init__(self):
        self.base_name = self.name

    def populate(self):
        Serializable.__init__(self, self.dir_name)
        self.set_parameters()
        self.set_serialization_params()
        self.acquire_data()

        # discard multiple-source namings and restore correct config
        self.set_parameters()
        self.set_name()
        self.set_serialization_params()
        info("Restored semantic name to : {}".format(self.name))

    def get_vectors(self):
        return self.semantic_document_vectors

    def get_semantic_embeddings(self):
        return None

    def generate_vectors(self):
        if self.loaded_vectorized:
            debug("Skipping generating sem. vectors, since loaded vectorized data already.")
            return
        # map dicts to vectors
        with tictoc("Generation of [{}] semantic vectors".format(self.semantic_weights)):
            self.concept_order = sorted(self.reference_concepts)
            self.dimension = len(self.concept_order)

            if self.semantic_weights == "embeddings":
                error("Embedding information requires the context_embedding semantic disambiguation. It is {} instead.".format(
                    self.disambiguation), condition=self.disambiguation != "context_embedding")
                self.semantic_document_vectors = self.get_semantic_embeddings()
            elif self.semantic_weights in [defs.weights.bag, defs.weights.tfidf]:
                # get concept-wise frequencies
                # just use the bag class to extract dense
                bagtrain= Bag()
                # train
                self.semantic_document_vectors = bagtrain.get_dense(self.concept_freqs[0])
                self.semantic_vector_indices = [np.arange(len(self.semantic_document_vectors))]
                # test
                if self.concept_freqs[1].shape[0] > 0:
                    bagtest = Bag()
                    test_features = bagtest.get_dense(self.concept_freqs[1])
                    self.semantic_document_vectors = np.append(self.semantic_document_vectors, test_features, axis=0)
                    self.semantic_vector_indices.append(np.arange(len(test_features), len(self.semantic_document_vectors)))
                else:
                    self.semantic_vector_indices.append(np.arange(0))
            else:
                error("Unimplemented semantic vector method: {}.".format(self.semantic_weights))



            self.semantic_epi = [np.ones((len(ind),), np.int32) for ind in self.semantic_vector_indices]
            write_pickled(self.serialization_path_vectorized, self.get_all_vectorized())

    # function to get a concept from a word, using the wordnet api
    # and a local word cache. Updates concept frequencies as well.
    def get_concept(self, word_information):
        if self.do_cache and word_information in self.lookup_cache:
            activations = self.lookup_cache[word_information]
        else:
            activations = self.lookup(word_information)
        if not activations:
            return {}
        if self.do_spread_activation:
            for concept in list(activations.keys()):
                hypers = self.run_spreading_activation(concept)
                for h in hypers:
                    activations[h] = hypers[h]
        if self.do_cache and word_information not in self.lookup_cache:
            # populate cache
            self.lookup_cache[word_information] = activations

        return activations

    def run_spreading_activation(self, concept):
        ret = {}
        # current weight value
        current_concepts = [concept]
        decay = 1
        for _ in range(self.spread_steps):
            decay *= self.spread_decay_factor
            new_concepts = []
            while current_concepts:
                concept = current_concepts.pop()
                if self.do_cache and concept in self.hypernym_cache:
                    hypers = self.hypernym_cache[concept]
                else:
                    hypers = self.spread_activation(concept)
                    if self.do_cache and concept not in self.hypernym_cache:
                        self.hypernym_cache[concept] = hypers
                if not hypers:
                    continue
                for h in hypers:
                    ret[h] = decay
                new_concepts.extend(hypers)
            current_concepts = new_concepts

        return ret

    # vectorized data handler
    def handle_vectorized(self, data):
        self.semantic_document_vectors, self.semantic_vector_indices, self.semantic_epi = \
         [data[k] for k in self.data_names_vectorized]
        self.loaded_preprocessed, self.loaded_vectorized = True, True
        debug(f"Read vectorized concept docs shape: {self.semantic_document_vectors.shape}, and epi: {shapes_list(self.semantic_epi)}")

    # preprocessed data getter
    def get_all_vectorized(self):
        return {k: v for (k,v) in zip(
            self.data_names_vectorized,
            [self.semantic_document_vectors, self.semantic_vector_indices, self.semantic_epi])
            }

    def get_all_preprocessed(self):
        return {"concept_weights": self.concept_freqs, "global_weights": self.global_freqs,
                "reference_concepts": self.reference_concepts}

    def lookup(self, candidate):
        error("Attempted to lookup from the base class")

    def set_parameters(self):
        if not is_none(self.config.limit):
            self.limit_type, self.limit_number = self.config.limit

        self.semantic_weights = self.config.weights
        # self.semantic_unit = self.config.unit

        self.disambiguation = self.config.disambiguation.lower()

        if self.config.spreading_activation:
            self.do_spread_activation = True
            self.spread_steps, self.spread_decay_factor = self.config.spreading_activation

        self.set_name()

    @staticmethod
    def generate_name(config, input_name=None):
        name_components = [config.name,
                           "w{}".format(config.weights),
                           "" if is_none(config.limit) else "".join(map(str, config.limit)),
                           "" if is_none(config.disambiguation) else "disam{}".format(config.disambiguation),
                           "" if is_none(config.spreading_activation) else "spread{}".format("-".join(map(str, config.spreading_activation)))
                           ]
        if input_name is not None and not config.misc.independent_component:
            # include the dataset in the sem. resource name
            name_components = [input_name] + name_components
        name_components = filter(lambda x: x, name_components)
        return "_".join(filter(lambda x: x != '', name_components))

    # make name string from components
    def set_name(self):
        self.name = self.generate_name(self.config, self.source_name)
        self.config.full_name = self.name

    # apply disambiguation to choose a single semantic unit from a collection of such
    def disambiguate(self, concepts, word_information, override=None):
        disam = self.disambiguation if not override else override
        if disam == defs.disam.first:
            return [concepts[0]]
        elif disam == defs.disam.pos:
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

    def get_cache_path(self):
        return join(self.config.folders.raw_data, self.dir_name, self.base_name + ".cache.pkl")

    def get_hypernym_cache_path(self):
        return join(self.config.folders.raw_data, self.dir_name, self.base_name + ".hypernym.cache.pkl")

    # read existing resource-wise serialized semantic cache from previous runs to speedup resolving
    def load_semantic_cache(self):
        if not self.do_cache:
            return None
        # direct cache
        cache_path = self.get_cache_path()
        if exists(cache_path):
            self.lookup_cache = read_pickled(cache_path)
            info("Read a {}-long semantic cache from {}.".format(len(self.lookup_cache), cache_path))
        # spreading activation cache
        hypernym_cache_path = self.get_cache_path()
        if exists(hypernym_cache_path):
            self.hypernym_cache = read_pickled(hypernym_cache_path)
            info("Read a {}-long hypernym semantic cache from {}.".format(len(self.hypernym_cache), hypernym_cache_path))

    # write the semantic cache after resolution of the current dataset
    def write_semantic_cache(self):
        if not self.do_cache:
            return
        cache_path = self.get_cache_path()
        if not exists(dirname(cache_path)):
            makedirs(dirname(cache_path), exist_ok=True)
        info("Writing a {}-long semantic cache to {}.".format(len(self.lookup_cache), cache_path))
        write_pickled(cache_path, self.lookup_cache)

    def process_similar_loaded(self):
        """Handles the cases where similar but not exactly equal data has been loaded.

        Some prost-processing is required to arrive at the exact configuration (e.g. filtering or TFIDF scaling)"""
        # if loaded non-filtered data and limiting is applied
        if not is_none(self.config.limit):
            if self.limit_type == defs.limit.top and len(self.reference_concepts) < self.limit_number:
                # only left to filtering with the bag object
                self.bag = Bag()
                self.bag.populate_all_data(self.concept_freqs, self.global_freqs, self.reference_concepts)
                self.bag.set_term_filtering(self.limit_type, self.limit_number)
                self.bag.filter_terms()
                self.concept_freqs = self.bag.get_weights()
                self.global_freqs = self.bag.get_global_weights()
                self.reference_concepts = self.bag.get_term_list()

        if self.semantic_weights == defs.weights.tfidf:
            # loaded concept-wise and global-wise frequencies; only left to compute TFIDF
            info("Computing TFIDF from the loaded preprocessed raw frequencies.")
            # train
            self.bag = TFIDF()
            self.bag.idf_normalize((self.concept_freqs[0], self.global_freqs[0]))
            self.concept_freqs[0] = self.bag.get_weights()
            # test
            self.bag_test = TFIDF()
            self.bag_test.idf_normalize((self.concept_freqs[1], self.global_freqs[1]))
            self.concept_freqs[1] = self.bag_test.get_weights()

    # function to map words to wordnet concepts
    def map_text(self):
        if self.loaded_vectorized:
            debug("Skipping mapping text due to vectorized data already loaded.")
            return
        if self.loaded_preprocessed:
            if self.semantic_weights == defs.weights.bag:
                debug("Skipping mapping text due to preprocessed data already loaded.")
                return
            self.process_similar_loaded()
            return

        self.initialize_lookup()

        # compute bag from scratch
        bag_class = get_bag_class(self.semantic_weights)
        self.bag_train, self.bag_test =  bag_class(), bag_class()

        # read the semantic resource input-concept cache , if it exists
        self.load_semantic_cache()

        # get representation-relatd information of the data to semantically annotate, if applicable
        train_data, test_data = self.text

        # train
        info("Extracting {}-{} semantic information from the training dataset".format(self.name, self.semantic_weights))
        self.bag_train.set_term_weighting_function(self.get_concept)
        self.bag_train.set_term_delineation_function(self.get_term_delineation)
        if not is_none(self.config.limit):
            self.bag_train.set_term_filtering(self.limit_type, self.limit_number)
        self.bag_train.set_term_extraction_function(lambda x: x)
        self.bag_train.map_collection(train_data)
        self.reference_concepts = self.bag_train.get_term_list()

        # test - since we restrict to the training set concepts, no need to filter
        info("Extracting {}-{} semantic information with a {}-long term list from the test dataset".
             format(self.name, self.semantic_weights, len(self.reference_concepts)))
        self.bag_test.set_term_list(self.reference_concepts)
        self.bag_test.set_term_weighting_function(self.get_concept)
        self.bag_test.set_term_delineation_function(self.get_term_delineation)
        self.bag_test.set_term_extraction_function(lambda x: x)
        self.bag_test.map_collection(test_data)

        # collect vectors
        self.concept_freqs = [self.bag_train.get_weights(), self.bag_test.get_weights()]
        self.global_freqs = [self.bag_train.get_global_weights(), self.bag_test.get_global_weights()]

        # write results
        info("Writing semantic assignment results to {}.".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

        # store the cache
        self.write_semantic_cache()

    def spread_activation(self, synset):
        error("Attempted to call abstract spread activation for semantic resource {}.".format(self.name))

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.concept_freqs, self.global_freqs, self.reference_concepts = [preprocessed[x] for x in self.data_names]
        debug("Read preprocessed concept docs shapes: {}, {}".format(*shapes_list(self.concept_freqs)))

    def get_term_delineation(self, document_text):
        """ Function to produce a list of terms of interest, from which to extract concepts """
        # default: word information
        return [word_info for word_info in document_text]

    def initialize_lookup(self):
        pass

    # region: # chain methods
    def run(self):
        self.process_component_inputs()
        self.populate()
        self.map_text()
        self.generate_vectors()
        self.outputs.set_vectors(Numeric(vecs=self.semantic_document_vectors))
        self.outputs.set_indices(Indices(indices=self.semantic_vector_indices, epi=self.semantic_epi))

    def configure_name(self):
        self.source_name = self.inputs.get_source_name()
        self.set_parameters()
        self.set_name()
        Component.configure_name(self, self.name)

    def process_component_inputs(self):
        error("{} requires text.".format(self.get_full_name()), not self.inputs.has_text())
        self.text = self.inputs.get_text().instances

from os.path import join, exists, dirname
from os import makedirs
from utils import tictoc, error, info, debug, write_pickled, read_pickled, shapes_list, warning
from serializable import Serializable
from representation.bag import Bag, TFIDF
from defs import is_none
import copy
import defs

class SemanticResource(Serializable):
    dir_name = "semantic"
    semantic_name = None
    name = None
    do_spread_activation = False
    loaded_vectorized = False

    lookup_cache = {}
    word_concept_embedding_cache = {}

    concept_freqs = []
    concept_context_word_threshold = None

    reference_concepts = None

    disambiguation = None
    pos_tag_mapping = {}
    representation = None

    do_cache = True

    data_names = ["concept_weights", "global_weights", "reference_concepts"]

    @staticmethod
    def get_available():
        return [cls.name for cls in SemanticResource.__subclasses__()]

    def set_additional_serialization_sources(self):
        self.serialization_path_vectorized = self.serialization_path_preprocessed + ".vectorized"
        self.data_paths.insert(0, self.serialization_path_vectorized)
        self.read_functions.insert(0, read_pickled)
        self.handler_functions.insert(0, self.handle_vectorized)

    def set_multiple_config_names(self):
        semantic_names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.limit.none]
        if not is_none(self.config.semantic.limit):
            filter_vals.append(self.config.semantic.limit)
        # any combo of weights, since local and global freqs are stored
        weight_vals = defs.weights.avail

        for w in weight_vals:
            for f in filter_vals:
                conf = copy.deepcopy(self.config)
                conf.semantic.weights = w
                conf.semantic.limit = f
                candidate_name = self.generate_name(conf)
                debug("Semantic config candidate: {}".format(candidate_name))
                semantic_names.append(candidate_name)

        self.multiple_config_names = semantic_names

    def __init__(self):
        self.base_name = self.name
        if not self.config.misc.skip_deserialization:
            Serializable.__init__(self, self.dir_name)
            self.set_serialization_params()
        else:
            warning("Skipping deserialization for the semantic component.")
        self.set_parameters()

        if not self.config.misc.skip_deserialization:
            self.acquire_data()
        # restore correct config
        self.set_parameters()
        self.set_name()
        import ipdb; ipdb.set_trace()
        self.set_serialization_params()
        info("Restored semantic name to : {}".format(self.name))


    def get_vectors(self):
        return self.semantic_document_vectors

    def generate_vectors(self):
        if self.loaded_vectorized:
            info("Skipping generating, since loaded vectorized data already.")
            return
        if self.representation.loaded_enriched():
            info("Skipping generating, since loaded enriched data already.")
            return
        # map dicts to vectors
        with tictoc("Generation of [{}] semantic vectors".format(self.semantic_weights)):
            self.concept_order = sorted(self.reference_concepts)
            self.dimension = len(self.concept_order)

            if self.semantic_weights == "embeddings":
                error("Embedding information requires the context_embedding semantic disambiguation. It is {} instead.".format(
                    self.disambiguation), condition=self.disambiguation != "context_embedding")
                self.semantic_document_vectors = self.get_semantic_embeddings()
            elif self.semantic_weights in [defs.weights.frequencies, defs.weights.tfidf]:
                # get concept-wise frequencies
                bagtrain, bagtest = Bag(), Bag()
                self.semantic_document_vectors = bagtrain.get_dense(self.concept_freqs[0]), bagtest.get_dense(self.concept_freqs[1])
            else:
                error("Unimplemented semantic vector method: {}.".format(self.semantic_weights))

            write_pickled(self.serialization_path_vectorized, [self.semantic_document_vectors, self.concept_order])

    # function to get a concept from a word, using the wordnet api
    # and a local word cache. Updates concept frequencies as well.
    def get_concept(self, word_information):
        if self.do_cache and word_information in self.lookup_cache:
            concept_activations = self.lookup_cache[word_information]
            # debug("Cache hit! for {}".format(word_information))
        else:
            concept_activations = self.lookup(word_information)
            if not concept_activations:
                return []
            if self.do_cache:
                # populate cache
                self.lookup_cache[word_information] = concept_activations
        return concept_activations

    # vectorized data handler
    def handle_vectorized(self, data):
        self.semantic_document_vectors, self.concept_order = data
        self.loaded_preprocessed = True
        self.loaded_vectorized = True
        debug("Read vectorized concept docs shapes: {}, {} and concept order: {}".format(*shapes_list(self.semantic_document_vectors), len(self.concept_order)))

    # preprocessed data getter
    def get_all_preprocessed(self):
        return {"concept_weights": self.concept_freqs, "global_weights": self.global_freqs,
                "reference_concepts": self.reference_concepts}

    def lookup(self, candidate):
        error("Attempted to lookup from the base class")

    def set_parameters(self):
        if not is_none(self.config.semantic.limit):
            self.limit_type, self.limit_number = self.config.semantic.limit

        self.semantic_weights = self.config.semantic.weights
        # self.semantic_unit = self.config.semantic.unit

        self.disambiguation = self.config.semantic.disambiguation.lower()

        if self.config.semantic.spreading_activation:
            self.do_spread_activation = True
            self.spread_steps, self.spread_decay_factor = self.config.semantic.spreading_activation

        self.set_name()

    @staticmethod
    def generate_name(config, include_dataset=True):
        if not config.has_semantic():
            return None
        name_components = [config.semantic.name,
                           "w{}".format(config.semantic.weights),
                           "".join(map(str,config.semantic.limit)),
                           "" if is_none(config.semantic.disambiguation) else "disam{}".format(config.semantic.disambiguation),
                           "" if is_none(config.semantic.spreading_activation) else "_spread{}".format("-".join(map(str,config.semantic.spreading_activation)))
                           ]
        if include_dataset and not config.misc.independent_component:
            # include the dataset in the sem. resource name
            name_components = [config.dataset
                                   .full_name] + name_components
        return "_".join(filter(lambda x: x != '', name_components))

    # make name string from components
    def set_name(self):
        self.name = self.generate_name(self.config)
        self.config.semantic.full_name = self.name

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
        return join(self.config.folders.raw_data, self.dir_name, self.base_name + ".cache.pickle")

    # read existing resource-wise serialized semantic cache from previous runs to speedup resolving
    def load_semantic_cache(self):
        if not self.do_cache:
            return None
        cache_path = self.get_cache_path()
        if exists(cache_path):
            self.lookup_cache = read_pickled(cache_path)
            info("Read a {}-long semantic cache from {}.".format(len(self.lookup_cache), cache_path))
            return self.lookup_cache
        return {}

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
        if not is_none(self.config.semantic.limit):
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
    def map_text(self, representation, dataset):
        self.representation = representation
        if self.representation.loaded_enriched():
            info("Skipping mapping text due to enriched data already loaded.")
            return
        if self.loaded_vectorized:
            info("Skipping mapping text due to vectorized data already loaded.")
            return
        if self.loaded_preprocessed:
            if self.semantic_weights == defs.weights.frequencies:
                info("Skipping mapping text due to preprocessed data already loaded.")
                return
            self.process_similar_loaded()
            return

        # compute bag from scratch
        if self.semantic_weights == defs.weights.tfidf:
            self.bag_train, self.bag_test = TFIDF(), TFIDF()
        else:
            self.bag_train, self.bag_test = Bag(), Bag()

        # read the semantic resource input-concept cache , if it exists
        self.load_semantic_cache()

        # process the dataset
        # train
        info("Extracting {}-{} semantic information from the training dataset".format(self.name, self.semantic_weights))
        self.bag_train.set_term_weighting_function(self.get_concept)
        self.bag_train.set_term_delineation_function(self.get_term_delineation)
        if not is_none(self.config.semantic.limit):
            self.bag_train.set_term_filtering(self.limit_type, self.limit_number)
        self.bag_train.set_term_extraction_function(lambda x: x)
        self.bag_train.map_collection(dataset.train)
        self.reference_concepts = self.bag_train.get_term_list()

        # test - since we restrict to the training set concepts, no need to filter
        info("Extracting {}-{} semantic information with a {}-long term list from the test dataset".format(self.name, self.semantic_weights, len(self.reference_concepts)))
        self.bag_test.set_term_list(self.reference_concepts)
        self.bag_test.set_term_weighting_function(self.get_concept)
        self.bag_test.set_term_delineation_function(self.get_term_delineation)
        self.bag_test.set_term_extraction_function(lambda x: x)
        self.bag_test.map_collection(dataset.test)

        # collect vectors
        self.concept_freqs = [self.bag_train.get_weights(), self.bag_test.get_weights()]
        self.global_freqs = [self.bag_train.get_global_weights(), self.bag_test.get_global_weights()]

        # write results
        info("Writing semantic assignment results to {}.".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

        # store the cache
        self.write_semantic_cache()

    def spread_activation(self, synset, steps_to_go, current_decay):
        error("Attempted to call abstract spread activation for semantic resource {}.".format(self.name))

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.concept_freqs, self.global_freqs, self.reference_concepts = [preprocessed[x] for x in self.data_names]
        debug("Read preprocessed concept docs shapes: {}, {}".format(*shapes_list(self.concept_freqs)))

    def get_term_delineation(self, document_text):
        """ Function to produce a list of terms of interest, from which to extract concepts """
        # default: word information
        return [word_info for word_info in document_text]


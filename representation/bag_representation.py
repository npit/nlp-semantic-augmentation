import copy
from os.path import basename

import numpy as np

import defs
from defs import is_none
from representation.bag import Bag
from representation.representation import Representation
from utils import debug, error, info, read_lines, shapes_list, write_pickled
from bundle.datatypes import Text


class BagRepresentation(Representation):
    name = "bag"
    term_list = None
    ngram_range = None

    data_names = Representation.data_names + ["term_list"]

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        # check if a dimension meta file exists, to read dimension
        Representation.__init__(self)

    def set_params(self):
        if self.config.ngram_range is not None:
            self.ngram_range = self.config.ngram_range

        if self.config.term_list is not None:
            self.read_term_list()
        self.compatible_aggregations = [defs.alias.none]
        self.compatible_sequence_lengths = [defs.sequence_length.unit]
        Representation.set_params(self)

    def read_term_list(self):
        try:
            with open(self.config.term_list) as f:
                self.term_list = [x.strip() for x in f.readlines()]
        except:
            error("Term list for bag should be a newline-delimited file, one term per line.")

    @staticmethod
    def generate_name(config, input_name):
        name = Representation.generate_name(config, input_name)
        if config.dimension is not None:
            if config.max_terms is not None:
                name += str(config.max_terms)
        return name

    # def set_multiple_config_names(self):
    #     """
    #     Declare which configurations match the current one as suitable to be loaded
    #     """
    #     return
    #     names = []
    #     # + no filtering, if filtered is specified
    #     filter_vals = [defs.limit.none]
    #     if self.do_limit:
    #         filter_vals.append(self.config.limit)
    #     # any combo of weights, since they're all stored
    #     weight_vals = defs.weights.avail
    #     for w in weight_vals:
    #         for f in filter_vals:
    #             conf = copy.deepcopy(self.config)
    #             conf.name = w
    #             conf.limit = f
    #             candidate_name = self.generate_name(conf, self.source_name)
    #             names.append(candidate_name)
    #             debug("Bag config candidate: {}".format(candidate_name))
    #     self.multiple_config_names = names


    def set_name(self):
        # disable the dimension
        Representation.set_name(self)
        # if external term list, add its length to the name
        if self.config.term_list is not None:
            self.name += "_tok_{}".format(basename(self.config.term_list))
        if self.config.dimension is None:
            # get max-terms name info if not already defined the dim
            if self.config.max_terms is not None:
                self.name += str(self.config.max_terms)
        else:
            # make sure dim and max-terms number match
            if self.config.dimension != self.config.max_terms:
                error(f"Specified different dimension and max terms: {self.config.dimension}, {self.config.max_terms}")

    def set_resources(self):
        if self.config.term_list is not None:
            self.resource_paths.append(self.config.term_list)
            self.resource_read_functions.append(read_lines)
            self.resource_handler_functions.append(self.handle_term_list)
            self.resource_always_load_flag.append(False)

    def handle_term_list(self, tok_list):
        info("Using external, {}-length term list.".format(len(tok_list)))
        self.term_list = tok_list

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        res = super().get_all_preprocessed()
        res["term_list"] = self.term_list
        return res

    # sparse to dense

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        # intead of undefined word index, get the term list
        self.term_list = preprocessed["term_list"]
        super().handle_preprocessed(preprocessed)
        # set misc required variables
        self.set_constant_elements_per_instance()

    def handle_aggregated(self, data):
        self.handle_preprocessed()
        self.loaded_aggregated = True
        # peek vector dimension
        data_dim = len(self.embeddings)
        if self.dimension is not None:
            if self.dimension != data_dim:
                error("Configuration for {} set to dimension {} but read {} from data.".format(self.name, self.dimension, data_dim))
        self.dimension = data_dim

    def load_model_from_disk(self):
        """Load bag model from disk"""
        if super().load_model():
            self.term_list = self.model
            return True
        return False

    def get_model(self):
        return self.term_list

    def build_model_from_inputs(self):
        """Build the bag model"""
        if self.term_list is None:
            # no supplied token list -- use vocabulary of the training dataset
            # self.term_list = self.vocabulary
            # info("Setting bag dimension to {} from input vocabulary.".format(len(self.term_list)))
            # will generate the vocabulary from the input
            pass
        info(f"Building {self.name} model")
        bagger = None
        if self.config.max_terms is not None:
            bagger = Bag(vocabulary=self.term_list, weighting=self.base_name, ngram_range=self.ngram_range, max_terms=self.config.max_terms)
        else:
            bagger = Bag(vocabulary=self.term_list, weighting=self.base_name, ngram_range=self.ngram_range)

        train_idx = self.indices.get_train_instances()
        texts = Text.get_strings(self.text.data.get_slice(train_idx))
        bagger.map_collection(texts, fit=True, transform=False)
        self.term_list = bagger.get_vocabulary()

        self.dimension = len(self.term_list)
        self.config.dimension = self.dimension



    def produce_outputs(self):
        """Map text to bag representations"""
        # if self.loaded_aggregated:
        #     debug("Skippping {} mapping due to preloading".format(self.base_name))
        #     return
        # need to calc term numeric index for aggregation


        # if self.loaded_preprocessed:
        #     debug("Skippping {} mapping due to preloading".format(self.base_name))
        #     return

        bagger = Bag(vocabulary=self.term_list, weighting=self.base_name, ngram_range=self.ngram_range)

        self.embeddings = np.ndarray((0, len(self.term_list)), dtype=np.int32)
        for idx in self.indices.get_train_test():
            texts = Text.get_strings(self.text.data.get_slice(idx))
            vecs = bagger.map_collection(texts, fit=False, transform=True)
            self.embeddings = np.append(self.embeddings, vecs, axis=0)
            del texts

        # texts = Text.get_strings(self.text.data.get_slice(test_idx))
        # vec_test = bagger.map_collection(texts, fit=do_fit)
        # del texts

        # self.embeddings = np.vstack((vec_train, vec_test))

        # self.embeddings = np.append(vec_train, vec_test)
        # self.vector_indices = (np.arange(len(train)), np.arange(len(test)))

        # set misc required variables
        self.set_constant_elements_per_instance()

class TFIDFRepresentation(BagRepresentation):
    name = "tfidf"

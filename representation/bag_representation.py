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
    do_limit = None
    ngram_range = None

    data_names = Representation.data_names + ["term_list"]

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        # check if a dimension meta file exists, to read dimension
        Representation.__init__(self)

    def set_params(self):
        # bag term limiting
        self.do_limit = False
        if not is_none(self.config.limit):
            self.do_limit = True
            self.limit_type, self.limit_number = self.config.limit
        if self.config.ngram_range is not None:
            self.ngram_range = self.config.ngram_range
        self.compatible_aggregations = [defs.alias.none]
        self.compatible_sequence_lengths = [defs.sequence_length.unit]
        Representation.set_params(self)

    @staticmethod
    def generate_name(config, input_name):
        name = Representation.generate_name(config, input_name)
        name_components = []
        if config.limit is not defs.limit.none:
            name_components.append("".join(map(str, config.limit)))
        return name + "_".join(name_components)

    def set_multiple_config_names(self):
        """
        Declare which configurations match the current one as suitable to be loaded
        """
        names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.limit.none]
        if self.do_limit:
            filter_vals.append(self.config.limit)
        # any combo of weights, since they're all stored
        weight_vals = defs.weights.avail
        for w in weight_vals:
            for f in filter_vals:
                conf = copy.deepcopy(self.config)
                conf.name = w
                conf.limit = f
                candidate_name = self.generate_name(conf, self.source_name)
                names.append(candidate_name)
                debug("Bag config candidate: {}".format(candidate_name))
        self.multiple_config_names = names

    def set_name(self):
        # disable the dimension
        Representation.set_name(self)
        # if external term list, add its length to the name
        if self.config.term_list is not None:
            self.name += "_tok_{}".format(basename(self.config.term_list))
        if not defs.is_none(self.config.limit):
            self.name += "".join(map(str, self.config.limit))

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
        return {"vector_indices": self.indices.instances, "elements_per_instance": self.indices.elements_per_instance,
                "term_list": self.term_list, "embeddings": self.embeddings, "roles": self.indices.roles}

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
            self.term_list = self.vocabulary
            info("Setting bag dimension to {} from input vocabulary.".format(len(self.term_list)))
        self.dimension = self.limit_number if self.do_limit else len(self.term_list)
        self.config.dimension = self.dimension

        bagger = Bag(vocabulary=self.term_list, weighting=self.base_name, ngram_range=self.ngram_range)
        if self.do_limit:
            bagger.set_thresholds(self.limit_type, self.limit_number)

        train_idx = self.indices.get_train_instances()
        texts = Text.get_strings(self.text.data.get_instance(train_idx))
        bagger.map_collection(texts, fit=True, transform=False)
        self.term_list = bagger.get_vocabulary()


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
            texts = Text.get_strings(self.text.data.get_instance(idx))
            vecs = bagger.map_collection(texts, fit=False, transform=True)
            self.embeddings = np.append(self.embeddings, vecs, axis=0)
            del texts

        # texts = Text.get_strings(self.text.data.get_instance(test_idx))
        # vec_test = bagger.map_collection(texts, fit=do_fit)
        # del texts

        # self.embeddings = np.vstack((vec_train, vec_test))

        # self.embeddings = np.append(vec_train, vec_test)
        # self.vector_indices = (np.arange(len(train)), np.arange(len(test)))

        # set misc required variables
        self.set_constant_elements_per_instance()

class TFIDFRepresentation(BagRepresentation):
    name = "tfidf"
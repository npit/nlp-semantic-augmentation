import copy
from os.path import basename

import numpy as np

import defs
from defs import is_none
from representation.bag import TFIDF, Bag
from representation.representation import Representation
from utils import debug, error, info, read_lines, shapes_list, write_pickled


class BagRepresentation(Representation):
    name = "bag"
    bag_class = Bag
    term_list = None
    do_limit = None

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
        return {"vector_indices": self.vector_indices, "elements_per_instance": self.elements_per_instance,
                "term_list": self.term_list, "embeddings": self.embeddings, "global_weights": self.global_weights}

    # sparse to dense

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        # intead of undefined word index, get the term list
        self.global_weights = preprocessed["global_weights"]
        self.term_list = preprocessed["term_list"]
        super().handle_preprocessed(preprocessed)
        # set misc required variables
        self.set_constant_elements_per_instance()

    def handle_aggregated(self, data):
        self.handle_preprocessed()
        self.loaded_aggregated = True
        # peek vector dimension
        data_dim = len(self.vector_indices[0][0])
        if self.dimension is not None:
            if self.dimension != data_dim:
                error("Configuration for {} set to dimension {} but read {} from data.".format(self.name, self.dimension, data_dim))
        self.dimension = data_dim

    def get_model(self):
        return self.term_list

    def map_text(self):
        if self.loaded_aggregated:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return
        # need to calc term numeric index for aggregation
        if self.term_list is None:
            # no supplied token list -- use vocabulary of the training dataset
            self.term_list = self.vocabulary
            info("Setting bag dimension to {} from input vocabulary.".format(len(self.term_list)))
        if self.do_limit:
            self.dimension = self.limit_number
        else:
            self.dimension = len(self.term_list)
        self.config.dimension = self.dimension
        # self.accomodate_dimension_change()
        # info("Renamed representation after bag computation to: {}".format(self.name))

        # calc term index mapping
        self.term_index = {term: self.term_list.index(term) for term in self.term_list}

        # if self.dimension is not None and self.dimension != len(self.term_list):
        #     error("Specified an explicit bag dimension of {} but term list contains {} elements (delete it?).".format(self.dimension, len(self.term_list)))

        if self.loaded_preprocessed:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return

        train, test = self.text
        # train
        self.bag_train = self.bag_class()
        self.bag_train.set_term_list(self.term_list)
        if self.do_limit:
            self.bag_train.set_term_filtering(self.limit_type, self.limit_number)
        self.bag_train.map_collection(train)
        self.global_weights = self.bag_train.global_weights
        
        if self.do_limit:
            self.term_list = self.bag_train.get_term_list()
            self.term_index = {k: v for (k, v) in self.term_index.items() if k in self.term_list}

        # # set representation dim and update name
        # self.dimension = len(self.term_list)
        # self.accomodate_dimension_change()

        # test
        self.bag_test = self.bag_class()
        self.bag_test.set_term_list(self.term_list)
        self.bag_test.map_collection(test)

        self.vector_indices = (np.arange(len(train)), np.arange(len(test)))

        # set misc required variables
        self.set_constant_elements_per_instance()

        self.embeddings = np.vstack((self.bag_train.get_dense(), self.bag_test.get_dense()))


        # write mapped data
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

        # if the representation length is not preset, write a small file with the dimension
        if self.config.term_list is not None:
            with open(self.serialization_path_preprocessed + ".meta", "w") as f:
                f.write(str(self.dimension))


class TFIDFRepresentation(BagRepresentation):
    name = "tfidf"
    bag_class = TFIDF

    def __init__(self, config):
        BagRepresentation.__init__(self, config)

    # nothing to load, can be computed on the fly
    def fetch_raw(self, path):
        pass

    def handle_preprocessed(self, preprocessed):
        super().handle_preprocessed(preprocessed)
        file_loaded_from = basename(self.successfully_loaded_path)
        if file_loaded_from.startswith(BagRepresentation.name):
            # apply IDF normalization
            self.embeddings = TFIDF.idf_normalize_dense(self.embeddings, self.global_weights)
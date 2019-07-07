import copy
from os.path import basename

import defs
from representation.bag import TFIDF, Bag
from representation.representation import Representation, is_none
from utils import debug, read_lines, info, shapes_list, error, write_pickled


class BagRepresentation(Representation):
    name = "bag"
    bag_class = Bag
    term_list = None
    do_limit = None

    data_names = ["dataset_vectors", "elements_per_instance", "term_list"]

    def __init__(self, config):
        self.config = config
        # check if a dimension meta file exists, to read dimension
        Representation.__init__(self)

    def set_params(self):
        # bag term limiting
        self.do_limit = False
        if not is_none(self.config.representation.limit):
            self.do_limit = True
            self.limit_type, self.limit_number = self.config.representation.limit
        self.compatible_aggregations = [defs.alias.none]
        self.compatible_sequence_lengths = [defs.sequence_length.unit]
        Representation.set_params(self)

    @staticmethod
    def generate_name(config, input_name):
        name = Representation.generate_name(config, input_name)
        name_components = []
        if config.representation.limit is not defs.limit.none:
            name_components.append("".join(map(str, config.representation.limit)))
        return name + "_".join(name_components)

    def set_multiple_config_names(self):
        """
        Declare which configurations match the current one as suitable to be loaded
        """
        names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.limit.none]
        if self.do_limit:
            filter_vals.append(self.config.representation.limit)
        # any combo of weights, since they're all stored
        weight_vals = defs.weights.avail
        for w in weight_vals:
            for f in filter_vals:
                conf = copy.deepcopy(self.config)
                conf.representation.name = w
                conf.representation.limit = f
                candidate_name = self.generate_name(conf)
                names.append(candidate_name)
                debug("Bag config candidate: {}".format(candidate_name))
        self.multiple_config_names = names

    def set_name(self):
        # disable the dimension
        Representation.set_name(self)
        # if external term list, add its length to the name
        if self.config.representation.term_list is not None:
            self.name += "_tok_{}".format(basename(self.config.representation.term_list))
        if not defs.is_none(self.config.representation.limit):
            self.name += "".join(map(str, self.config.representation.limit))

    def set_resources(self):
        if self.config.representation.term_list is not None:
            self.resource_paths.append(self.config.representation.term_list)
            self.resource_read_functions.append(read_lines)
            self.resource_handler_functions.append(self.handle_term_list)
            self.resource_always_load_flag.append(False)

    def handle_term_list(self, tok_list):
        info("Using external, {}-length term list.".format(len(tok_list)))
        self.term_list = tok_list

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        return {"dataset_vectors": self.dataset_vectors, "elements_per_instance": self.elements_per_instance,
                "term_list": self.term_list}

    # sparse to dense
    def compute_dense(self):
        if self.loaded_transformed:
            debug("Will not compute dense, since transformed data were loaded")
            return
        info("Computing dense representation for the bag.")
        self.dataset_vectors = [self.bag_train.get_dense(), self.bag_test.get_dense()]
        info("Computed dense dataset shapes: {} {}".format(*shapes_list(self.dataset_vectors)))

    def aggregate_instance_vectors(self):
        # bag representations produce a single instance-level vectors
        if self.aggregation != defs.alias.none:
            error("Specified {} aggregation with {} representation, but only {} is compatible.".format(self.aggregation, self.name, defs.alias.none))
        pass

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        # intead of undefined word index, get the term list
        self.dataset_vectors, self.dataset_words, self.term_list = [preprocessed[n] for n in self.data_names]
        # set misc required variables
        self.elements_per_instance = [[1 for _ in ds] for ds in self.dataset_vectors]

    def handle_aggregated(self, data):
        self.handle_preprocessed()
        self.loaded_aggregated = True
        # peek vector dimension
        data_dim = len(self.dataset_vectors[0][0])
        if self.dimension is not None:
            if self.dimension != data_dim:
                error("Configuration for {} set to dimension {} but read {} from data.".format(self.name, self.dimension, data_dim))
        self.dimension = data_dim

    def set_transform(self, transform):
        """Update representation information as per the input transform"""
        self.name += transform.get_name()
        self.dimension = transform.get_dimension()

        data = transform.get_all_preprocessed()
        self.dataset_vectors, self.elements_per_instance, self.term_list = [data[n] for n in self.data_names]
        self.loaded_transformed = True

    def accomodate_dimension_change(self):
        self.set_params()
        # the superclass method above reads the dimension from the config -- set from the Bag field
        self.set_name()
        self.set_serialization_params()
        self.set_additional_serialization_sources()

    def map_text(self, dset):
        if  self.loaded_aggregated:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return
        # need to calc term numeric index for aggregation
        if self.term_list is None:
            # no supplied token list -- use vocabulary of the training dataset
            self.term_list = dset.vocabulary
            info("Setting bag dimension to {} from dataset vocabulary.".format(len(self.term_list)))
        if self.do_limit:
            self.dimension = self.limit_number
        else:
            self.dimension = len(self.term_list)
        self.config.representation.dimension = self.dimension
        self.accomodate_dimension_change()
        info("Renamed representation after bag computation to: {}".format(self.name))

        # calc term index mapping
        self.term_index = {term: self.term_list.index(term) for term in self.term_list}

        # if self.dimension is not None and self.dimension != len(self.term_list):
        #     error("Specified an explicit bag dimension of {} but term list contains {} elements (delete it?).".format(self.dimension, len(self.term_list)))

        if self.loaded_preprocessed:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return
        info("Mapping {} to {} representation.".format(dset.name, self.name))

        self.dataset_words = [self.term_list, None]
        self.dataset_vectors = []

        # train
        self.bag_train = self.bag_class()
        self.bag_train.set_term_list(self.term_list)
        if self.do_limit:
            self.bag_train.set_term_filtering(self.limit_type, self.limit_number)
        self.bag_train.map_collection(dset.train)
        self.dataset_vectors.append(self.bag_train.get_weights())
        if self.do_limit:
            self.term_list = self.bag_train.get_term_list()
            self.term_index = {k: v for (k, v) in self.term_index.items() if k in self.term_list}

        # # set representation dim and update name
        # self.dimension = len(self.term_list)
        # self.accomodate_dimension_change()

        # test
        self.bag_test = self.bag_class()
        self.bag_test.set_term_list(self.term_list)
        self.bag_test.map_collection(dset.test)
        self.dataset_vectors.append(self.bag_test.get_weights())

        # set misc required variables
        self.set_constant_elements_per_instance()

        # write mapped data
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

        # if the representation length is not preset, write a small file with the dimension
        if self.config.representation.term_list is not None:
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


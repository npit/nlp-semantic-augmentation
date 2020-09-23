import numpy as np

import defs
from bundle.datatypes import *
from bundle.datausages import *
from component.component import Component
from serializable import Serializable
from utils import debug, error, info, shapes_list, set_constant_epi


class Representation(Serializable):
    component_name = "representation"
    dir_name = "representation"
    compatible_aggregations = []
    compatible_sequence_lengths = []
    sequence_length = 1
    vector_indices = None

    data_names = ["vector_indices", "elements_per_instance", "embeddings"]
    consumes = Text.name
    produces = Numeric.name

    @staticmethod
    def get_available():
        return [cls.name for cls in Representation.__subclasses__()]

    def __init__(self):
        """Constructor"""
        pass

    def populate(self):
        Serializable.__init__(self, self.dir_name)
        # check for serialized mapped data
        self.set_serialization_params()
        # set required resources
        self.set_resources()
        # fetch the required data
        self.acquire_data()
        # restore name, maybe
        if self.multiple_config_names:
            self.configure_name()
            self.set_serialization_params()
            # self.set_params()
            # self.set_name()
            # Component.configure_name(self)
            # self.check_params()
            info("Restored representation name to {}".format(self.name))

    # region # serializable overrides

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.vector_indices, self.elements_per_instance, self.embeddings = [preprocessed[n] for n in Representation.data_names]
        debug("Read preprocessed dataset embeddings shapes: {}".format(shapes_list(self.vector_indices)))

    # add exra representations-specific serialization paths
    def set_additional_serialization_sources(self):
        # compute names
        aggr = "".join(list(map(str, [self.config.aggregation] + [self.sequence_length])))
        self.serialization_path_aggregated = "{}/{}.aggregated_{}.pkl".format(self.serialization_dir, self.name, aggr)
        self.add_serialization_source(self.serialization_path_aggregated, handler=self.handle_aggregated)

    def handle_aggregated(self, data):
        self.handle_preprocessed(data)
        self.loaded_aggregated = True
        debug("Read aggregated embeddings shapes: {}, {}".format(*shapes_list(self.vector_indices)))

    def handle_raw(self, raw_data):
        pass

    def fetch_raw(self, path):
        # assume embeddings are dataframes
        return None

    def preprocess(self):
        pass

    # endregion

    # region # getter functions
    def get_zero_pad_element(self):
        return np.zeros((1, self.dimension), np.float32)

    def get_data(self):
        return self.vector_indices

    def get_name(self):
        return self.name

    def get_dimension(self):
        return self.dimension

    def get_vectors(self):
        return self.vector_indices

    def get_elements_per_instance(self):
        return self.elements_per_instance
    # endregion

    def set_params(self):
        """Set baseline compatibilities and read relevant config values"""
        self.compatible_aggregations = [defs.alias.none, None]
        self.compatible_sequence_lengths = [defs.sequence_length.unit]

        self.aggregation = self.config.aggregation
        self.dimension = self.config.dimension
        self.dataset_name = self.source_name

        self.sequence_length = self.config.sequence_length

    def check_params(self):
        if self.aggregation not in self.compatible_aggregations:
            error("[{}] aggregation incompatible with {}. Compatible ones are: {}!".format(self.aggregation, self.base_name, self.compatible_aggregations))
        error("Unset sequence length compatibility for representation {}".format(self.base_name), not self.compatible_sequence_lengths)
        if defs.get_sequence_length_type(self.sequence_length) not in self.compatible_sequence_lengths:
            error("Incompatible sequence length {} with {}. Compatibles are {}".format(
                self.sequence_length, self.base_name, self.compatible_sequence_lengths))

    @staticmethod
    def generate_name(config, input_name):
        return "{}_{}_dim{}".format(config.name, input_name, config.dimension)

    # name setter function, exists for potential overriding
    def set_name(self):
        self.name = Representation.generate_name(self.config, self.source_name)

    def set_constant_elements_per_instance(self, num=1):
        """Function to assign single-element (default) instances"""
        if not self.vector_indices:
            error("Attempted to set constant epi before computing dataset vectors.")
        self.elements_per_instance = [set_constant_epi(ds) for ds in self.vector_indices]

    def set_identity_indexes(self, data):
        """Function to assign unique indexes to data"""
        self.vector_indices = [np.arange(len(d)) for d in data]

    # data getter for semantic processing
    def process_data_for_semantic_processing(self, train, test):
        return train, test

    # abstracts
    def map_text(self):
        error("Attempted to access abstract function map_text of {}".format(self.name))

    # region  # chain methods

    def configure_name(self):
        self.set_params()
        self.set_name()
        self.check_params()
        Component.configure_name(self, self.name)

    def run(self):
        self.populate()
        self.process_component_inputs()
        self.map_text()

        # set outputs
        vectors = Numeric(self.embeddings)
        dp = DataPack(vectors, self.indices, self.name)
        self.data_pool.add_data(dp)

    def process_component_inputs(self):
        # if self.loaded_aggregated or self.loaded_preprocessed:
        #     return
        error("{} requires a text input.".format(self.name), not self.data_pool.has_text())
        text = self.data_pool.request_data(Text.name, Indices.name, self.name)
        self.text, self.vocabulary = text.data.instances, text.data.vocabulary
        self.indices = text.get_usage(Indices.name)

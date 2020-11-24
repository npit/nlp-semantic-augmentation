"""Module for feature transformation methods
This module provides methods that transform existing representations into others.
"""

import numpy as np

from bundle.bundle import DataPool
from bundle.datatypes import *
from bundle.datausages import *
from component.component import Component
from defs import roles
from serializable import Serializable
from utils import (debug, error, info, match_labels_to_instances, shapes_list,
                   write_pickled, read_pickled)


class Transform(Serializable):
    """Abstract transform class

    """
    component_name = "transform"
    name = None
    dimension = None
    dir_name = "transform"
    do_reinitialize = None
    transformer = None
    process_func_train = None
    process_func_test = None
    is_supervised = False
    term_components = None

    produces=Numeric.name
    consumes=Numeric.name

    @staticmethod
    def get_available():
        return [cls.base_name for cls in Transform.__subclasses__()]

    def __init__(self, config):
        if self.is_supervised:
            self.consumes = [Numeric.name, Labels.name]
        self.name = self.base_name
        self.config = config
        self.dimension = config.dimension

    def populate(self):
        Serializable.__init__(self, self.dir_name)
        self.set_serialization_params()
        self.acquire_data()

    def get_dimension(self):
        return self.dimension

    def get_name(self):
        return self.name

    def get_model(self):
        return self.transformer

    def compute(self):
        """Apply transform on input features"""
        if self.loaded():
            debug("Skipping {} computation due to transform data already loaded." .format(self.name))
            return

        # sanity checks
        error("Got transform dimension of {} but input dimension is {}.".format(self.dimension, self.input_dimension), self.input_dimension < self.dimension)

        # output containers
        self.term_components = []

        # compute
        info("Applying {} {}-dimensional transform on the raw representation.".
             format(self.base_name, self.dimension))

        # train
        train_data = self.input_vectors[self.train_index, :]

        info("Transforming training input data shape: {}".format(train_data.shape))
        if self.is_supervised:
            ground_truth = np.reshape(match_labels_to_instances(self.train_epi, self.train_labels), (len(train_data), ))
            self.vectors = self.process_func_train(train_data, ground_truth)
        else:
            self.vectors = self.process_func_train(train_data)
        self.output_roles = (roles.train,)

        if self.test_index.size > 0:
            # make zero output matrix
            output_data = np.zeros((len(self.input_vectors), self.dimension), np.float32)
            output_data[self.train_index, :] = self.vectors

            test_data = self.input_vectors[self.test_index, :]
            info("Transforming test input data shape: {}".format(test_data.shape))
            vecs = self.process_func_test(test_data)
            output_data[self.test_index, :] = vecs
            self.vectors = output_data

            self.output_roles = (roles.train, roles.test)
        else:
            info(f"Skipping empty test indexes.")

        self.term_components = self.get_term_representations()
        self.verify_transformed(self.vectors)
        info(f"Output shape: {self.vectors.shape}")
        # write the output data
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())
        # write the trained transformer model
        self.save_model()

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        return self.vectors

    def handle_preprocessed(self, data):
        self.vectors = data

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.components_

    def verify_transformed(self, data):
        """Checks if the projected data dimension matches the one prescribed
        """
        data_dim = data.shape[-1]
        if data_dim != self.dimension:
            error(
                "{} result dimension {} does not match the prescribed input dimension {}"
                .format(self.name, data_dim, self.dimension))
        nans, _ = np.where(np.isnan(data))
        if np.size(nans) != 0:
            error("{} result contains nan elements in :{}".format(
                self.name, nans))

    def configure_name(self):
        if self.is_supervised:
            # set source name at the vector source
            error(
                "{} is supervised and needs an input bundle list.".format(
                    self.get_full_name()),
                len(self.inputs) <= 1)
            # get the longest name, most probable
            print(self.inputs.get_source_name())
            self.input_name = max(self.inputs.get_source_name(),
                                  key=lambda x: len(x))
        else:
            error(
                "{} is not supervised but got an input bundle list, instead of a single bundle."
                .format(self.get_full_name()),
                len(self.inputs) <= 1)
            # get the longest name, most probable
            self.input_name = self.inputs.get_source_name()
        self.name = "{}_{}_{}".format(self.input_name, self.base_name,
                                      self.dimension)
        Component.configure_name(self, self.name)

    def run(self):
        """Transform component run function"""
        error("{} needs vector information.".format(self.get_full_name()), not self.inputs.has_vectors())

        if self.is_supervised:
            error("{} is supervised and needs an input bundle list.".format(self.get_full_name()),
                  len(self.inputs) <= 1)
            error("{} got a {}-long bundle list, but a length of at most 2 is required."
                .format(self.get_full_name(), len(self.inputs)),
                len(self.inputs) > 2)
            # verify that the input name is of the vectors source bundle
            vectors_bundle_index = [
                i for i in range(len(self.inputs))
                if self.inputs.get(i).has_vectors()
            ][0]
            expected_name = self.inputs.get(
                vectors_bundle_index).get_source_name()
            if self.input_name != expected_name:
                error("Supervised transform was configured to name {} but the vector bundle {} is encountered at runtime."
                    .format(self.input_name, expected_name))

            self.input_vectors = self.inputs.get(vectors_bundle_index).get_vectors().instances
            self.train_epi = self.inputs.get(vectors_bundle_index).get_indices().elements_per_instance
            error(f"{self.get_full_name()} is supervised and needs label information.", not self.inputs.has_labels())
            self.train_labels = self.inputs.get_labels(enforce_single=True, roles=roles.train)
        else:
            error("{} is not supervised but got an input bundle list, instead of a single bundle."
                .format(self.get_full_name()), len(self.inputs) <= 1)
            self.input_vectors = self.inputs.get_vectors().instances

        # indexes
        self.train_index = self.inputs.get_indices(role=roles.train)
        self.test_index = self.inputs.get_indices(role=roles.test)

        self.populate()
        self.input_dimension = self.input_vectors[0].shape[-1]
        self.compute()
        # set the outputs: transformed vectors and identical indices
        self.outputs.set_vectors(Numeric(vecs=self.vectors))
        self.outputs.set_indices(self.inputs.get_indices())

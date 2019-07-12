from bundle.bundle import BundleList, Bundle
from bundle.datatypes import Vectors, Labels
from component.component import Component
from utils import error, info, shapes_list, write_pickled, match_labels_to_instances, debug
import numpy as np
from serializable import Serializable

"""Module for feature transformation methods

This module provides methods that transform existing representations into others.
"""


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


    @staticmethod
    def get_available():
        return [cls.base_name for cls in Transform.__subclasses__()]

    def __init__(self, config):
        Component.__init__(self, produces=Vectors.name, consumes=Vectors.name)
        if self.is_supervised:
            self.consumes = [Vectors.name, Labels.name]
        self.name = self.base_name
        self.config = config
        self.dimension = config.transform.dimension

    def populate(self):
        Serializable.__init__(self, self.dir_name)
        self.set_serialization_params()
        self.acquire_data()

    def get_dimension(self):
        return self.dimension

    def get_name(self):
        return self.name

    def compute(self):
        """Apply transform on input features"""
        if self.loaded():
            debug("Skipping {} computation due to transform data already loaded.".format(self.name))
            return

        # sanity checks
        error("Got transform dimension of {} but input dimension is {}.".format(self.dimension, self.input_dimension), self.input_dimension < self.dimension)

        # output containers
        self.vectors = []
        self.term_components = []

        # compute
        num_chunks = len(self.input_vectors)
        info("Applying {} {}-dimensional transform on the raw representation.".format(self.base_name, self.dimension))
        for v, vecs in enumerate(self.input_vectors):
            if v == 0:
                info("Transforming training input data shape {}/{}: {}".format(v + 1, num_chunks, vecs.shape))
                if self.is_supervised:
                    ground_truth = np.reshape(match_labels_to_instances(self.elements_per_instance[v], self.labels[v]), (len(vecs),))
                    self.vectors.append(self.process_func_train(vecs, ground_truth))
                else:
                    self.vectors.append(self.process_func_train(vecs))
            else:
                info("Transforming test input data shape {}/{}: {}".format(v + 1, num_chunks, vecs.shape))
                self.vectors.append(self.process_func_test(vecs))
            self.verify_transformed(self.vectors[-1])
            self.term_components.append(self.get_term_representations())

        info("Output shapes (train/test): {}, {}".format(*shapes_list(self.vectors)))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

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
            error("{} result dimension {} does not match the prescribed input dimension {}".format(self.name, data_dim, self.dimension))
        nans = np.where(np.isnan(data))
        if np.size(nans) != 0:
            error("{} result contains nan elements in :{}".format(self.name, nans))

    def configure_name(self):
        if self.is_supervised:
            # set source name at the vector source
            error("{} is supervised and needs an input bundle list.".format(self.get_full_name()), type(self.inputs) is not BundleList)
            # get the longest name, most probable
            self.input_name = max(self.inputs.get_source_name())
        else:
            error("{} is not supervised but got an input bundle list, instead of a single bundle.".format(self.get_full_name()), type(self.inputs) is BundleList)
            # get the longest name, most probable
            self.input_name = self.inputs.get_source_name()
        self.name = "{}_{}_{}".format(self.input_name, self.base_name, self.dimension)

    def run(self):
        error("{} needs vector information.".format(self.get_full_name()), not self.inputs.has_vectors())
        if self.is_supervised:
            error("{} is supervised and needs an input bundle list.".format(self.get_full_name()), type(self.inputs) is not BundleList)
            error("{} got a {}-long bundle list, but a length of at most 2 is required.".format(self.get_full_name(), len(self.inputs)), len(self.inputs) > 2)
            # verify that the input name is of the vectors source bundle
            vectors_bundle_index = [i for i in range(len(self.inputs)) if self.inputs.get(i).has_vectors()][0]
            expected_name = self.inputs.get(vectors_bundle_index).get_source_name()
            if self.input_name != expected_name:
                error("Supervised transform was configured to name {} but the vector bundle {} is encountered at runtime.".format(self.input_name, expected_name))

            self.input_vectors = self.inputs.get(vectors_bundle_index).get_vectors().instances
            self.elements_per_instance = self.inputs.get(vectors_bundle_index).get_vectors().elements_per_instance
            error("{} is supervised and needs label information.".format(self.get_full_name()), not self.inputs.has_labels())
            self.labels = self.inputs.get_labels(single=True)
        else:
            error("{} is not supervised but got an input bundle list, instead of a single bundle.".format(self.get_full_name()), type(self.inputs) is BundleList)
            self.input_vectors = self.inputs.get_vectors().instances
            self.elements_per_instance = self.inputs.get_vectors().elements_per_instance
        self.populate()
        self.input_dimension = self.input_vectors[0].shape[-1]
        self.compute()
        self.outputs.set_vectors(Vectors(vecs=self.vectors))

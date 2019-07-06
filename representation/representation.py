from os.path import basename, isfile
import pandas as pd

from component.bundle import Bundle
from utils import error, tictoc, info, debug, write_pickled, warning, shapes_list, read_lines, one_hot, get_shape
import numpy as np
from serializable import Serializable
from semantic.semantic_resource import SemanticResource
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from representation.bag import Bag, TFIDF
from defs import *
import defs
import copy


class Representation(Serializable):
    component_name = "representation"
    dir_name = "representation"
    loaded_transformed = False
    compatible_aggregations = []
    compatible_sequence_lengths = []
    sequence_length = 1

    data_names = ["dataset_vectors", "elements_per_instance"]

    @staticmethod
    def get_available():
        return [cls.name for cls in Representation.__subclasses__()]

    def __init__(self, can_fail_loading=True):
        """Constructor"""
        pass

    def populate(self):
        self.set_params()
        self.set_name()
        Serializable.__init__(self, self.dir_name)
        # check for serialized mapped data
        self.set_serialization_params()
        # add paths for aggregated / transformed / enriched representations:
        # set required resources
        self.set_resources()
        # fetch the required data
        self.acquire_data()


    # region # serializable overrides

    # add exra representations-specific serialization paths
    def set_additional_serialization_sources(self):
        # compute names
        aggr = "".join(list(map(str, [self.config.representation.aggregation] + [self.sequence_length])))
        self.serialization_path_aggregated = "{}/{}.aggregated_{}.pickle".format(self.serialization_dir, self.name, aggr)
        self.add_serialization_source(self.serialization_path_aggregated, handler=self.handle_aggregated)

    def handle_aggregated(self, data):
        self.handle_preprocessed(data)
        self.loaded_aggregated = True
        debug("Read aggregated embeddings shapes: {}, {}".format(*shapes_list(self.dataset_vectors)))

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

    def get_vocabulary_size(self):
        return len(self.dataset_words[0])

    def get_data(self):
        return self.dataset_vectors

    def get_name(self):
        return self.name

    def get_dimension(self):
        return self.dimension

    def get_vectors(self):
        return self.dataset_vectors

    def get_elements_per_instance(self):
        return self.elements_per_instance
    # endregion

    # shortcut for reading configuration values
    def set_params(self):
        self.aggregation = self.config.representation.aggregation
        if self.aggregation not in self.compatible_aggregations:
            error("{} aggregation incompatible with {}. Compatible ones are: {}!".format(self.aggregation, self.base_name, self.compatible_aggregations))

        self.dimension = self.config.representation.dimension
        self.dataset_name = self.inputs.get_source_name()
        self.base_name = self.name

        self.sequence_length = self.config.representation.sequence_length
        error("Unset sequence length compatibility for representation {}".format(self.base_name), not self.compatible_sequence_lengths)
        if defs.get_sequence_length_type(self.sequence_length) not in self.compatible_sequence_lengths:
            error("Incompatible sequence length {} with {}. Compatibles are {}".format(
                self.sequence_length, self.base_name, self.compatible_sequence_lengths))

    @staticmethod
    def generate_name(config, input_name):
        return "{}_{}_dim{}".format(config.representation.name, input_name, config.representation.dimension)

    # name setter function, exists for potential overriding
    def set_name(self):
        self.name = Representation.generate_name(self.config, self.inputs.get_source_name())

    # finalize embeddings to use for training, aggregating all data to a single ndarray
    # if semantic enrichment is selected, do the infusion
    def set_semantic(self, semantic):
        """
        Attach semantic component to the representation
        :param semantic: the dense semantic vectors
        """
        if self.config.semantic.enrichment is not None:
            semantic_data = semantic.get_vectors()
            info("Enriching [{}] embeddings with shapes: {} {} and {} vecs/doc with [{}] semantic information of shapes {} {}.".
                 format(self.config.representation.name, *shapes_list(self.dataset_vectors), self.sequence_length,
                        self.config.semantic.name, *shapes_list(semantic_data)))

            if self.config.semantic.enrichment == "concat":
                semantic_dim = len(semantic_data[0][0])
                final_dim = self.dimension + semantic_dim
                for dset_idx in range(len(semantic_data)):
                    info("Concatenating dataset part {}/{} to composite dimension: {}".format(dset_idx + 1, len(semantic_data), final_dim))
                    if self.sequence_length > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.sequence_length)),
                                                             (-1, semantic_dim))
                    self.dataset_vectors[dset_idx] = np.concatenate(
                        [self.dataset_vectors[dset_idx], semantic_data[dset_idx]], axis=1)

            elif self.config.semantic.enrichment == "replace":
                final_dim = len(semantic_data[0][0])
                for dset_idx in range(len(semantic_data)):
                    info("Replacing dataset part {}/{} with semantic info of dimension: {}".format(dset_idx + 1, len(semantic_data), final_dim))
                    if self.sequence_length > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.sequence_length)),
                                                             (-1, final_dim))
                    self.dataset_vectors[dset_idx] = semantic_data[dset_idx]
            else:
                error("Undefined semantic enrichment: {}".format(self.config.semantic.enrichment))

            # serialize finalized embeddings
            self.dimension = final_dim
            write_pickled(self.serialization_path_finalized, self.get_all_preprocessed())

    def has_word(self, word):
        return word in self.embeddings.index

    def match_labels_to_instances(self, dset_idx, gt, do_flatten=True, binarize_num_labels=None):
        """Expand, if needed, ground truth samples for multi-vector instances
        """
        epi = self.elements_per_instance[dset_idx]
        multi_vector_instance_idx = [i for i in range(len(epi)) if epi[i] > 1]
        if not multi_vector_instance_idx:
            if binarize_num_labels is not None:
                return one_hot(gt, num_labels=binarize_num_labels)
            return gt
        res = []
        for i in range(len(gt)):
            # get the number of elements for the instance
            times = epi[i]
            if do_flatten:
                res.extend([gt[i] for _ in range(times)])
            else:
                res.append([gt[i] for _ in range(times)])

        if binarize_num_labels is not None:
            return one_hot(res, num_labels=binarize_num_labels)
        return res

    # set one element per instance
    def set_constant_elements_per_instance(self, num=1):
        if not self.dataset_vectors:
            error("Attempted to set constant epi before computing dataset vectors.")
        self.elements_per_instance = [[num for _ in ds] for ds in self.dataset_vectors]

    # data getter for semantic processing
    def process_data_for_semantic_processing(self, train, test):
        return train, test

    # abstracts
    def aggregate_instance_vectors(self):
        error("Attempted to run abstract aggregate_instance_vectors for {}".format(self.name))

    def compute_dense(self):
        error("Attempted to run abstract compute_dense for {}".format(self.name))

    def __str__(self):
        return self.name

    def map_text(self):
        error("Attempted to access abstract function map_text of {}".format(self.name))

    # region  # chain methods
    def get_outputs(self):
        # yield mapped dataset vectors
        return Bundle(self.name, vectors=self.dataset_vectors, labels=self.inputs.get_labels())

    def run(self):
        self.populate()
        self.map_text()
        self.compute_dense()
        self.aggregate_instance_vectors()



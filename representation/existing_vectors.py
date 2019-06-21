"""Class to handle loading already extracted features to be evaluated in the learning pipeline.
The expected format is <dataset_name>.existing.csv
"""
from os.path import isfile, basename

import pandas as pd

from representation.embedding import Embedding
from utils import error, info, write_pickled


class ExistingVectors(Embedding):
    name = "existing"

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.representation.name
        Embedding.__init__(self)

    def set_resources(self):
        if self.loaded():
            return
        dataset_name = self.config.dataset.name
        if isfile(dataset_name):
            dataset_name = basename(dataset_name)
        csv_mapping_name = "{}/{}.{}.csv".format(self.raw_data_dir, dataset_name, self.base_name)
        self.resource_paths.append(csv_mapping_name)
        self.resource_read_functions.append(self.read_vectors)
        self.resource_handler_functions.append(lambda x: x)

    # read the vectors the dataset is mapped to
    def read_vectors(self, path):
        self.vectors = pd.read_csv(path, index_col=None, header=None, sep=self.config.misc.csv_separator).values
        error("Malformed existing vectors loaded: {}".format(self.vectors.shape), not all(self.vectors.shape))
        self.dimension = self.vectors.shape[-1]

    def map_text(self, dset):
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping dataset: {} to {} feature vectors.".format(dset.name, self.name))
        if self.sequence_length * (len(dset.train) + len(dset.test)) != len(self.vectors):
            error("Loaded {} existing vectors for a seqlen of {} with a dataset of {} and {} train/test samples.". \
                  format(len(self.vectors), self.sequence_length, len(dset.train), len(dset.test)))
        self.dataset_vectors = [self.vectors[:self.sequence_length * len(dset.train)], self.vectors[:self.sequence_length * len(dset.test)]]
        self.set_constant_elements_per_instance(num=self.sequence_length)
        self.present_term_indexes = []
        del self.vectors
        # write
        info("Writing dataset mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

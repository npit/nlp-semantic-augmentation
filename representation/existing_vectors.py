from os.path import basename, isfile

import pandas as pd

from representation.embedding import Embedding
from utils import error, info, write_pickled


class ExistingVectors(Embedding):
    """Class to handle loading already extracted features.
    The expected format is <dataset_name>.existing.csv
    """
    name = "existing"

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.name
        Embedding.__init__(self)

    def set_resources(self):
        if self.loaded():
            return
        dataset_name = self.inputs.source_name
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

    def map_text(self):
        if self.loaded_preprocessed or self.loaded_aggregated:
            return
        dset = self.inputs.get_indices()
        train_idx = dset.instances[dset.get_train_role_indexes()[0]]
        test_idx = dset.instances[dset.get_test_role_indexes()[0]]
        num_train, num_test = len(train_idx), len(test_idx)
        num_total = num_train + num_test
        info("Mapping dataset: {} to {} feature vectors.".format(self.inputs.source_name, self.name))
        if self.sequence_length * num_total != len(self.vectors):
            error("Loaded {} existing vectors for a seqlen of {} with a dataset of {} and {} train/test (total {}) samples.". \
                  format(len(self.vectors), self.sequence_length, num_train, num_test, num_total))
        self.vector_indices = [train_idx, test_idx]
        self.set_constant_elements_per_instance(num=self.sequence_length)
        self.embeddings = self.vectors
        # write
        info("Writing dataset mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

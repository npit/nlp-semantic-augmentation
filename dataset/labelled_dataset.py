"""
Abstract class for labelled datasets.
"""

import random

import numpy as np

from bundle.datatypes import Labels
from dataset.dataset import Dataset
from utils import (align_index, debug, error, flatten, info, nltk_download,
                   tictoc, warning, write_pickled)


class LabelledDataset(Dataset):
    """
    Class encapsulating labelled dataset information
    """
    data_names = ["train-labels", "train-label-names", "test-labels", "test_label-names"]
    train_labels, test_labels = None, None
    multilabel = False

    def __init__(self):
        super().__init__()

    def check_sanity(self):
        """Perform label-related sanity checks on the dataset"""
        # ensure numeric labels
        try:
            for labels in [self.train_labels, self.test_labels]:
                list(map(int, flatten(labels)))
        except ValueError as ve:
            error("Non-numeric label encountered: {}".format(ve))
        except TypeError as ve:
            warning("Non-collection labelitem encountered: {}".format(ve))


    # region # getters

    def is_multilabel(self):
        """Multilabel status getter"""
        return self.multilabel

    def get_labels(self):
        """Labels getter"""
        return self.train_labels, self.test_labels

    def get_num_labels(self):
        """Number of labels getter"""
        return self.num_labels

    def get_info(self):
        """Current data information getter"""
        return super().get_info() + " {} labels".format(self.num_labels)

    def get_all_raw(self):
        """Raw data fetcher that combine parent vector data with labels"""
        data = super().get_all_raw()
        for key, value in zip(self.data_names, [self.train_labels, self.train_label_names, self.test_labels, self.test_label_names]):
            data[key] = value
        return data
    # endregion

    # region handlers
    def handle_raw_serialized(self, deserialized_data):
        self.train_labels, self.train_label_names, self.test_labels, self.test_label_names = \
            [deserialized_data[n] for n in self.data_names]
        self.num_labels = len(set(self.train_label_names))
        super().handle_raw_serialized(deserialized_data)
    # endregion


    # region # chain methods
    def set_outputs(self):
        """Set label data to the output bundle"""
        super().set_outputs()
        self.outputs.set_labels(Labels((self.train_labels, self.test_labels)))

    # endregion

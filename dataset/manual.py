"""Module for manual dataset specification"""
import json
from os.path import basename

import numpy as np

from dataset.dataset import Dataset
from dataset.manual_reader import ManualDatasetReader
from utils import write_pickled


class ManualDataset(Dataset):
    """Class to represent a custom dataset"""

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = basename(config.name)
        Dataset.__init__(self)

    def get_all_raw(self):
        data = Dataset.get_all_raw(self)
        data["language"] = self.language
        data["multilabel"] = self.multilabel
        return data

    # raw path getter
    def get_raw_path(self):
        return self.config.name

    def fetch_raw(self, raw_data_path):
        # cannot read a raw limited dataset
        if self.name != self.base_name:
            return None

        with open(raw_data_path) as f:
            raw_data = json.load(f)
        return raw_data

    def handle_raw(self, raw_data):
        mdr = ManualDatasetReader()
        mdr.read_dataset(raw_data=raw_data)
        self.train, self.test = mdr.train, mdr.test
        self.train_labels, self.test_labels = mdr.train_labels, mdr.test_labels
        self.multilabel = mdr.max_num_instance_labels > 1
        self.labelset, self.label_names = mdr.labelset, mdr.label_names
        self.language = mdr.language
        self.roles = mdr.roles

        # write serialized data
        write_pickled(self.serialization_path, self.get_all_raw())

    def handle_raw_serialized(self, deserialized_data):
        Dataset.handle_raw_serialized(self, deserialized_data)
        self.language = deserialized_data["language"]
        if self.train_labels is not None:
            self.multilabel = deserialized_data["multilabel"]
            self.labelset = sorted(np.unique(np.concatenate(self.train_labels)))

    def handle_serialized(self, deserialized_data):
        self.handle_raw_serialized(deserialized_data)

    def handle_preprocessed(self, deserialized_data):
        Dataset.handle_preprocessed(self, deserialized_data)

    def get_name(self):
        # get only the filename
        return basename(self.name)

    def get_base_name(self):
        # get only the base filename
        return basename(self.base_name)

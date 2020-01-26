from os.path import join

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from dataset.dataset import Dataset
from utils import error, info, write_pickled


class TwentyNewsGroups(Dataset):
    name = "20newsgroups"
    language = "english"

    def fetch_raw(self, dummy_input):
        # only applicable for raw dataset
        if self.name != self.base_name:
            return None
        info("Downloading {} via sklearn".format(self.name))

        train = fetch_20newsgroups(data_home=join(self.config.folders.raw_data, "sklearn"), subset='train')
        test = fetch_20newsgroups(data_home=join(self.config.folders.raw_data, "sklearn"), subset='test')
        return [train, test]

    def handle_raw(self, raw_data):

        # results are sklearn bunches
        # map to train/test/categories
        train, test = raw_data
        info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        self.train, self.test = train.data, test.data
        self.train_labels, self.test_labels = (np.split(x, len(x)) for x in (train.target, test.target))
        # self.train_labels, self.test_labels = np.array_split(train.target, len(train.target)), np.array_split(test.target, len(test.target))
        if not train.target_names == test.target_names:
            error("Non-matching label names for train and test set! {train.target_names} {test.target_names}")
        self.label_names = train.target_names
        self.labelset = list(sorted(set(train.target)))
        self.num_labels = len(self.label_names)
        # write serialized data
        write_pickled(self.serialization_path, self.get_all_raw())
        self.loaded_raw = True

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        Dataset.__init__(self)
        self.multilabel = False

    # raw path setter
    def get_raw_path(self):
        # dataset is downloadable
        pass

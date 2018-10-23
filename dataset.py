import logging
import numpy as np
import pickle
import os
from helpers import Config
from utils import error, tic, toc, info, warning
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords, reuters



class Dataset:
    name = ""
    limited_name = ""
    serialization_dir = "serialization/datasets"
    preprocessed = False
    train, test = None, None


    def load_preprocessed(self):
        if os.path.exists(self.serialization_path_preprocessed):
            info("Loading preprocessed {} dataset from {}.".format(self.name, self.serialization_path_preprocessed))
            with open(self.serialization_path_preprocessed, "rb") as f:
                self.train, self.train_target, self.train_label_names, self.test, self.test_target, self.test_label_names = pickle.load(f)
                self.num_labels = len(self.train_label_names)
                self.preprocessed = True
                info("Loaded {} train and {} test data, with {} labels".format(len(self.train), len(self.test), self.num_labels))
                return True
        return False


    def serialize_raw_dataset(self, data):
        info("Writing {} raw dataset to serialization path {}".format(self.name, self.serialization_path))
        with open(self.serialization_path, "wb") as f:
                pickle.dump(data, f)

    def suspend_limit(self):
        self.set_paths(self.name)

    def set_paths(self, name):
        if not os.path.exists(self.serialization_dir):
            os.makedirs(self.serialization_dir, exist_ok=True)
        self.serialization_path = "{}/raw_{}.pickle".format(self.serialization_dir, name)
        self.serialization_path_preprocessed = "{}/{}.preprocessed.pickle".format(self.serialization_dir, name)

    def __init__(self, name):
        self.name = name
        self.set_paths(name)

    def make(self, config):
        self.serialization_dir = os.path.join(config.get_serialization_dir(), "datasets")
        self.set_paths(self.name)
        info("Loaded {} train and {} test documents with {} labels for {}".format(len(self.train), len(self.test), self.num_labels, self.name))

    # data getter
    def get_data(self):
        return self.train, self.test

    def get_targets(self):
        return self.train_target, self.test_target

    def get_num_labels(self):
        return self.num_labels


    def apply_limit(self, config):
        if config.option("data_limit"):
            value = config.option("data_limit")
            if self.train:
                self.train = self.train[:value]
                self.test = self.test[:value]
                self.train_target = self.train_target[:value]
                self.test_target = self.test_target[:value]
                info("Limiting {} to {} items.".format(self.name, value))
            self.limited_name = self.name + "_limited_" + str(value)
            self.set_paths(self.limited_name)

    def preprocess(self):
        stopw = set(stopwords.words('english'))
        if self.preprocessed:
            info("Skipping preprocessing, loading existing data from {}.".format(self.serialization_path_preprocessed))
            return
        tic()
        filter = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
        info("Preprocessing {}".format(self.name))
        info("Mapping training set to word sequence.")
        for i in range(len(self.train)):
            processed = text_to_word_sequence(self.train[i], filters=filter, lower=True, split=' ')
            processed = [p for p in processed if p not in stopw]
            self.train[i] = processed
        info("Mapping test set to word sequence.")
        for i in range(len(self.test)):
            processed = text_to_word_sequence(self.test[i], filters=filter, lower=True, split=' ')
            processed = [p for p in processed if p not in stopw]
            self.test[i] = processed
        toc("Preprocessing")
        # serialize
        with open(self.serialization_path_preprocessed, "wb") as f:
            pickle.dump([self.train, self.train_target, self.train_label_names,
                         self.test, self.test_target, self.test_label_names], f)


    def get_name(self):
        return self.name


class TwentyNewsGroups(Dataset):
    name = "20newsgroups"
    language = "english"

    def __init__(self):
        Dataset.__init__(self, TwentyNewsGroups.name)

    # load dataset previously downloaded and serialized
    def load_serialized_dataset(self):
        info("Loading raw {} from serialization path {}".format(self.name, self.serialization_path))
        with open(self.serialization_path, "rb") as f:
            deser = pickle.load(f)
            train, test = deser[0], deser[1]
        return train, test

    # download and serialize dataset
    def download_raw_dataset(self):
        info("Downloading {} via sklearn".format(self.name))
        seed = Config().get_seed()
        train = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
        test = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
        # write
        self.serialize_raw_dataset([train, test])
        return train, test

    # prepare dataset
    def make(self, config):
        # enforce possible data num restrictions
        self.apply_limit(config)
        # if preprocessed data already exists, load them
        if self.load_preprocessed():
            return
        # enforce possible data num restrictions
        self.suspend_limit()

        # else, check if the downloaded & serialized raw data exists
        if os.path.exists(self.serialization_path) and os.path.isfile(self.serialization_path):
            train, test = self.load_serialized_dataset()
            pass
        else:
            # else, fetch from scikit-learn
            train, test = self.download_raw_dataset()

        # results are sklearn bunches
        info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        # map to train/test/categories
        self.train, self.test = train.data, test.data
        self.train_target, self.test_target = train.target, test.target
        self.train_label_names = train.target_names
        self.test_label_names = test.target_names
        self.num_labels = len(self.train_label_names)

        self.apply_limit(config)
        Dataset.make(self, config)

class Brown:
    pass

class Reuters(Dataset):
    name = "reuters"
    language = "english"

    def __init__(self):
        Dataset.__init__(self, Reuters.name)

    def load_serialized_dataset(self):
        info("Loading raw {} from serialization path {}".format(self.name, self.serialization_path))
        with open(self.serialization_path, "rb") as f:
            deser = pickle.load(f)
            self.train, self.train_target, self.test, self.test_target, self.num_labels, self.train_label_names, self.test_label_names = deser

    def download_raw_dataset(self):
        info("Downloading raw {} dataset".format(self.name))
        # get ids
        documents = reuters.fileids()
        train_ids = list(filter(lambda doc: doc.startswith("train"), documents))
        test_ids = list(filter(lambda doc: doc.startswith("test"), documents))
        categories = reuters.categories()
        self.num_labels = len(categories)
        self.train_label_names, self.test_label_names = set(), set()

        # get content
        self.train, self.test = [], []
        self.train_target, self.test_target = [], []
        for cat_index, cat in enumerate(categories):
            # get all docs in that category
            for doc in reuters.fileids(cat):
                # get its content
                content = reuters.raw(doc)
                # assign content
                if doc.startswith("training"):
                    self.train.append(content)
                    self.train_target.append(cat_index)
                    self.train_label_names.add(cat)
                else:
                    self.test.append(content)
                    self.test_target.append(cat_index)
                    self.test_label_names.add(cat)
        self.train_label_names = list(self.train_label_names)
        self.test_label_names = list(self.test_label_names)
        self.train_target = np.asarray(self.train_target, np.int32)
        self.test_target = np.asarray(self.train_target, np.int32)
        # serialize
        self.serialize_raw_dataset([self.train, self.train_target, self.test, self.test_target, self.num_labels, self.train_label_names, self.test_label_names])


    def make(self, config):

        # enforce possible data num restrictions
        self.apply_limit(config)
        # if preprocessed data already exists, load them
        if self.load_preprocessed():
            return
        # enforce possible data num restrictions
        self.suspend_limit()

        # else, check if the downloaded & serialized raw data exists
        if os.path.exists(self.serialization_path) and os.path.isfile(self.serialization_path):
            self.load_serialized_dataset()
        else:
            # else, fetch from nltk
            self.download_raw_dataset()


        self.apply_limit(config)
        Dataset.make(self, config)
    pass

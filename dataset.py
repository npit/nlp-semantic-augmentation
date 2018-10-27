import logging
import numpy as np
import pickle
import os
from os.path import exists, isfile, join
from os import makedirs
from helpers import Config
from utils import error, tic, toc, info, warning, read_pickled, write_pickled
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords, reuters

from serializable import Serializable



class Dataset(Serializable):
    name = ""
    vocabulary = set()
    vocabulary_index = []
    word_to_index = {}
    limited_name = ""
    serialization_subdir = "datasets"
    undefined_word_index = None
    preprocessed = False
    train, test = None, None

    def create(config):
        name = config.dataset.name
        if name == TwentyNewsGroups.name:
            return TwentyNewsGroups(config)
        elif name == Reuters.name:
            return Reuters(config)
        else:
            error("Undefined dataset: {}".format(name))


    # dataset creation
    def __init__(self, name = None):
        self.serialization_dir = join(self.config.folders.serialization, self.serialization_subdir)
        Serializable.__init__(self, self.serialization_dir)
        # if a limit is defined, check for such a serialized dataset
        if self.config.dataset.limit:
            self.apply_limit()
            res = self.acquire(fatal_error = False)
            if res:
                return
        self.suspend_limit()
        self.acquire(do_preprocess=False)
        # limit, if applicable
        self.apply_limit()
        self.preprocess()


    def handle_preprocessed(self, preprocessed):
        info("Loaded preprocessed {} dataset from {}.".format(self.name, self.serialization_path_preprocessed))
        self.train, self.train_target, self.train_label_names, \
        self.test, self.test_target, self.test_label_names, \
        self.vocabulary, self.vocabulary_index, self.undefined_word_index = preprocessed
        self.num_labels = len(self.train_label_names)
        self.preprocessed = True
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
        info("Loaded {} train and {} test data, with {} labels".format(len(self.train), len(self.test), self.num_labels))

    def suspend_limit(self):
        self.name = self.base_name

    # set paths according to current dataset name
    def set_paths(self, name):
        if not os.path.exists(self.serialization_dir):
            os.makedirs(self.serialization_dir, exist_ok=True)
        # raw dataset 
        self.serialization_path = "{}/raw_{}.pickle".format(self.serialization_dir, name)
        # preprocessed dataset
        self.serialization_path_preprocessed = "{}/{}.preprocessed.pickle".format(self.serialization_dir, name)

    def set_raw_path(self):
        error("Need to override raw path dataset setter for {}".format(self.name))

    def fetch_raw(self):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def load_serialized(self):
        error("Need to override serialized data loader for {}".format(self.name))

    def handle_raw_serialized(self, raw_serialized):
        error("Need to override raw serialized data handler for {}".format(self.name))

    # data getter
    def get_data(self):
        return self.train, self.test

    def get_targets(self):
        return self.train_target, self.test_target

    def get_num_labels(self):
        return self.num_labels

    def apply_limit(self):
        if self.config.dataset.limit:
            value = self.config.dataset.limit

            self.name = self.base_name + "_limited_" + str(value)
            self.set_paths(self.name)

            # if data has been loaded, limit the instances
            if self.train:
                self.train = self.train[:value]
                self.test = self.test[:value]
                self.train_target = self.train_target[:value]
                self.test_target = self.test_target[:value]
                info("Limited {} to {} items.".format(self.base_name, value))
                # serialize the limited version
                write_pickled(self.serialization_path, self.get_all_raw())

    # preprocess raw texts into word list
    def preprocess(self):
        stopw = set(stopwords.words(self.language))
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
            self.vocabulary.update(processed)
        info("Mapping test set to word sequence.")
        for i in range(len(self.test)):
            processed = text_to_word_sequence(self.test[i], filters=filter, lower=True, split=' ')
            processed = [p for p in processed if p not in stopw]
            self.test[i] = processed
        # fix word order and get word indexes
        self.vocabulary = list(self.vocabulary)
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
            self.vocabulary_index.append(index)
        # add another for the missing
        self.undefined_word_index = len(self.vocabulary)
        toc("Preprocessing")
        # serialize
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_all_raw(self):
        return [self.train, self.train_target, self.train_label_names,
                         self.test, self.test_target, self.test_label_names ]

    def get_all_preprocessed(self):
        return self.get_all_raw() + [self.vocabulary, self.vocabulary_index, self.undefined_word_index]


    def get_name(self):
        return self.name


class TwentyNewsGroups(Dataset):
    name = "20newsgroups"
    language = "english"

    def fetch_raw(self):
        info("Downloading {} via sklearn".format(self.name))
        seed = self.config.get_seed()
        train = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
        test = fetch_20newsgroups(subset='test', shuffle=True, random_state=seed)
        return [train, test]

    def handle_raw(self, raw_data):
        self.handle_raw_serialized(raw_data)

    def handle_raw_serialized(self, deserialized_data):
        train, test = deserialized_data

        # results are sklearn bunches
        info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        # map to train/test/categories
        self.train, self.test = train.data, test.data
        self.train_target, self.test_target = train.target, test.target
        self.train_label_names = train.target_names
        self.test_label_names = test.target_names
        self.num_labels = len(self.train_label_names)


    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        Dataset.__init__(self, TwentyNewsGroups.name)

    # raw path setter
    def set_raw_path(self):
        # dataset is downloadable
        pass



class Brown:
    pass

class Reuters(Dataset):
    name = "reuters"
    language = "english"

    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        Dataset.__init__(self, Reuters.name)

    def fetch_raw(self):
        # only applicable for raw dataset
        if self.name != self.base_name:
            return None
        info("Downloading raw {} dataset".format(self.name))
        # get ids
        documents = reuters.fileids()
        categories = reuters.categories()

        #train_ids = list(filter(lambda doc: doc.startswith("train"), documents))
        #test_ids = list(filter(lambda doc: doc.startswith("test"), documents))
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

        return [self.train, self.train_target, self.test, self.test_target,
                self.num_labels, self.train_label_names, self.test_label_names]

    def handle_raw(self, raw_data):
        # already processed
        pass

    def handle_raw_serialized(self, raw_serialized):
        self.train, self.train_target, self.test, self.test_target, \
        self.num_labels, self.train_label_names, self.test_label_names = raw_serialized
        pass

    # raw path setter
    def set_raw_path(self):
        # dataset is downloadable
        pass

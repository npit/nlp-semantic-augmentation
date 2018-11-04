import logging
import numpy as np
import pickle
import os
from os.path import exists, isfile, join
from os import makedirs
from helpers import Config
from utils import error, tictoc, info, warning, read_pickled, write_pickled
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords, reuters
import nltk

from serializable import Serializable



class Dataset(Serializable):
    name = ""
    vocabulary = set()
    vocabulary_index = []
    word_to_index = {}
    limited_name = ""
    dir_name = "datasets"
    undefined_word_index = None
    preprocessed = False
    train, test = None, None

    pos_tags = []

    def create(config):
        name = config.dataset.name
        if name == TwentyNewsGroups.name:
            return TwentyNewsGroups(config)
        elif name == Reuters.name:
            return Reuters(config)
        else:
            error("Undefined dataset: {}".format(name))


    # dataset creation
    def __init__(self):
        Serializable.__init__(self, self.dir_name)
        self.set_serialization_params()

        # check for limited dataset
        self.apply_limit()
        res = self.acquire2(fatal_error=False)
        if any(self.load_flags):
            # downloaded successfully
            self.loaded_index = self.load_flags.index(True)
        else:
            # check for raw dataset
            self.suspend_limit()
            # setup paths
            self.set_serialization_params()
            res = self.acquire2()
            self.loaded_index = self.load_flags.index(True)
        # limit, if applicable
        if not self.loaded_preprocessed:
            self.apply_limit()
            self.preprocess()


    def handle_preprocessed(self, preprocessed):
        info("Loaded preprocessed {} dataset from {}.".format(self.name, self.serialization_path_preprocessed))
        self.train, self.train_target, self.train_label_names, \
        self.test, self.test_target, self.test_label_names, \
        self.vocabulary, self.vocabulary_index, self.undefined_word_index, self.pos_tags = preprocessed

        self.num_labels = len(self.train_label_names)
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
        info("Loaded preprocessed data: {} train, {} test, with {} labels".format(len(self.train), len(self.test), self.num_labels))
        self.loaded_preprocessed = True

    def suspend_limit(self):
        self.name = self.base_name

    def get_raw_path(self):
        error("Need to override raw path datasea getter for {}".format(self.name))

    def fetch_raw(self):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

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
            self.set_paths_by_name(self.name)

            # if data has been loaded, limit the instances
            if self.train:
                self.train = self.train[:value]
                self.test = self.test[:value]
                self.train_target = self.train_target[:value]
                self.test_target = self.test_target[:value]
                info("Limited {} to {} items.".format(self.base_name, value))
                # serialize the limited version
                write_pickled(self.serialization_path, self.get_all_raw())

    def setup_nltk_resources(self):
        try:
            stopwords.words(self.language)
        except LookupError:
            nltk.download("stopwords")
        try:
            nltk.pos_tag("Text")
        except LookupError:
            nltk.download('averaged_perceptron_tagger')


    # map text string into list of stopword-filtered words and POS tags
    def process_single_text(self, text, filt, stopwords):
        words = text_to_word_sequence(text, filters=filt, lower=True, split=' ')
        pos_tags = nltk.pos_tag(words)
        # remove stopwords
        idx = [p for p in range(len(words)) if words[p] not in stopwords]
        words = [words[p] for p in idx]
        pos_tags = [pos_tags[p] for p in idx]
        return words, pos_tags

    # return POS information from the non-missing word indexes, per dataset and document
    def get_pos(self, present_word_idx):
        out_pos = []
        for dset in range(len(self.pos_tags)):
            out_pos.append([])
            for doc in range(len(self.pos_tags[dset])):
                pos = [self.pos_tags[dset][doc][i] for i in present_word_idx[dset][doc]]
                out_pos[-1].append(pos)
        return out_pos


    # preprocess raw texts into word list
    def preprocess(self):
        if self.loaded_preprocessed:
            info("Skipping preprocessing, preprocessed data already loaded from {}.".format(self.serialization_path_preprocessed))
            return
        self.setup_nltk_resources()

        stopw = set(stopwords.words(self.language))

        with tictoc("Preprocessing"):
            filt = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
            info("Preprocessing {}".format(self.name))
            info("Mapping training set to word sequences.")
            train_pos, test_pos = [],[]
            for i in range(len(self.train)):
                self.process_single_text(self.train[i], filt, stopw)
                words, pos_tags = self.process_single_text(self.train[i], filt=filt, stopwords=stopw)
                self.train[i] = words
                train_pos.append(pos_tags)
                self.vocabulary.update(words)
            info("Mapping test set to word sequences.")
            for i in range(len(self.test)):
                words, pos_tags = self.process_single_text(self.test[i], filt=filt, stopwords=stopw)
                self.test[i] = words
                test_pos.append(pos_tags)
            # set pos
            self.pos_tags = [train_pos, test_pos]
            # fix word order and get word indexes
            self.vocabulary = list(self.vocabulary)
            for index, word in enumerate(self.vocabulary):
                self.word_to_index[word] = index
                self.vocabulary_index.append(index)
            # add another for the missing
            self.undefined_word_index = len(self.vocabulary)

        # serialize
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_all_raw(self):
        return [self.train, self.train_target, self.train_label_names,
                         self.test, self.test_target, self.test_label_names ]

    def get_all_preprocessed(self):
        return self.get_all_raw() + [self.vocabulary, self.vocabulary_index, self.undefined_word_index, self.pos_tags]


    def get_name(self):
        return self.name


class TwentyNewsGroups(Dataset):
    name = "20newsgroups"
    language = "english"

    def fetch_raw(self, dummy_input):
        # only applicable for raw dataset
        if self.name != self.base_name:
            return None
        info("Downloading {} via sklearn".format(self.name))
        seed = self.config.get_seed()
        train = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
        test = fetch_20newsgroups(subset='test', shuffle=True, random_state=seed)
        return [train, test]

    def handle_raw(self, raw_data):

        # results are sklearn bunches
        # map to train/test/categories
        train, test = raw_data
        info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        self.train, self.test = train.data, test.data
        self.train_target, self.test_target = train.target, test.target
        self.train_label_names = train.target_names
        self.test_label_names = test.target_names
        self.num_labels = len(self.train_label_names)
        # write serialized data
        write_pickled(self.serialization_path, self.get_all_raw())
        self.loaded_raw = True

    def handle_raw_serialized(self, deserialized_data):
        self.train, self.train_target, self.train_label_names, \
            self.test, self.test_target, self.test_label_names  = deserialized_data
        self.num_labels = len(set(self.train_label_names))
        self.loaded_raw_serialized = True



    def __init__(self, config):
        self.config = config
        self.base_name = self.name
        Dataset.__init__(self)

    # raw path setter
    def get_raw_path(self):
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
        Dataset.__init__(self)

    def fetch_raw(self, dummy_input):
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
        return self.get_all_raw()

    def handle_raw(self, raw_data):
        # serialize
        write_pickled(self.serialization_path, raw_data)
        self.loaded_raw = True
        pass

    def handle_raw_serialized(self, raw_serialized):
        self.train, self.train_target, self.test, self.test_target, \
        self.num_labels, self.train_label_names, self.test_label_names = raw_serialized
        self.loaded_raw_serialized = True

    # raw path setter
    def get_raw_path(self):
        # dataset is downloadable
        pass

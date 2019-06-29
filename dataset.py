import random
import numpy as np
import pickle
import os
import json
from os.path import exists, isfile, join, basename
from os import makedirs
from helpers import Config
from nltk.tokenize import RegexpTokenizer
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
            return ManualDataset(config)


    # dataset creation
    def __init__(self):
        Serializable.__init__(self, self.dir_name)
        self.nltk_tokenizer = RegexpTokenizer(r'\w+')
        self.set_serialization_params()

        # check for limited dataset
        self.apply_limit()
        self.acquire2(fatal_error=False)
        if any(self.load_flags):
            # downloaded successfully
            self.loaded_index = self.load_flags.index(True)
        else:
            # check for raw dataset. Suspend limit and setup paths
            self.suspend_limit()
            self.set_serialization_params()
            # exclude loading of pre-processed data
            self.data_paths = self.data_paths[1:]
            self.read_functions = self.read_functions[1:]
            self.handler_functions = self.handler_functions[1:]
            # get the data but do not preprocess
            res = self.acquire2(do_preprocess=False)
            self.loaded_index = self.load_flags.index(True)
            # reapply the limit
            self.apply_limit()

        # if no preprocessed data was loaded, apply it now
        if not self.loaded_preprocessed:
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

    def handle_raw_serialized(self, deserialized_data):
        self.train, self.train_target, self.train_label_names, \
        self.test, self.test_target, self.test_label_names  = deserialized_data
        self.num_labels = len(set(self.train_label_names))
        self.loaded_raw_serialized = True



    # data getter
    def get_data(self):
        return self.train, self.test

    def get_targets(self):
        return self.train_target, self.test_target

    def get_num_labels(self):
        return self.num_labels

    # static method for external name computation
    def get_limited_name(config):
        name = basename(config.dataset.name)
        if config.has_class_limit():
            name += "_clim_{}".format(config.dataset.class_limit)
        if config.has_data_limit():
            ltrain, ltest = config.dataset.data_limit
            if ltrain:
                name += "_dlim_tr{}".format(ltrain)
            if ltest:
                name += "_dlim_te{}".format(ltest)
        return name

    def apply_data_limit(self, name):
        ltrain, ltest = self.config.dataset.data_limit
        if ltrain:
            name += "_dlim_tr{}".format(ltrain)
            if self.train:
                self.train = self.train[:ltrain]
                self.train_target = self.train_target[:ltrain]
                info("Limited {} loaded data to {} train items.".format(self.base_name, ltrain))
        if ltest:
            name += "_dlim_te{}".format(ltest)
            if self.test:
                self.test = self.test[:ltest]
                self.test_target = self.test_target[:ltest]
                info("Limited {} loaded data to {} test items.".format(self.base_name, ltest))
        return name

    def apply_class_limit(self, name):
        c_lim = self.config.dataset.class_limit
        if c_lim is not None:
            name += "_clim_{}".format(c_lim)
            if self.train:
                retained_classes = random.sample(list(range(self.num_labels)), c_lim)
                info("Limiting to classes: {}".format(retained_classes))
                data = [(x,y) for (x,y) in zip(self.train, self.train_target) if y in retained_classes]
                self.train, self.train_target = [list(x) for x in zip(*data)]
                data = [(x,y) for (x,y) in zip(self.test, self.test_target) if y in retained_classes]
                self.test, self.test_target = [list(x) for x in zip(*data)]
                self.num_labels = len(retained_classes)
                # remap retained classes to indexes starting from 0
                self.train_target = [retained_classes.index(rc) for rc in self.train_target]
                self.test_target = [retained_classes.index(rc) for rc in self.test_target]
                # fix the label names
                self.train_label_names = [self.train_label_names[rc] for rc in retained_classes]
                self.test_label_names = [self.test_label_names[rc] for rc in retained_classes]
                info("Limited {} dataset to {} classes: {} - i.e. {} - resulting in {} train and {} test data."\
                     .format(self.base_name, self.num_labels, retained_classes,
                             self.train_label_names, len(self.train), len(self.test)))
        return name

    def apply_limit(self):
        if self.config.has_limit():
            self.base_name = self.name
            name = self.apply_class_limit(self.base_name)
            self.name = self.apply_data_limit(name)
            self.set_paths_by_name(self.name)
        if self.train:
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
        # words = [w.lower() for w in self.nltk_tokenizer.tokenize(text)]
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


    def preprocess_single_chunk(self, document_list, track_vocabulary=False):
        self.setup_nltk_resources()
        filt = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
        stopw = set(stopwords.words(self.language))
        ret_words, ret_pos, ret_voc = [], [], set()
        num_words =[]
        for i in range(len(document_list)):
            words, pos_tags = self.process_single_text(document_list[i], filt=filt, stopwords=stopw)
            ret_words.append(words)
            ret_pos.append(pos_tags)
            if track_vocabulary:
                ret_voc.update(words)
            num_words.append(len(words))
        stats = [x(num_words) for x in [np.mean, np.var, np.std]]
        info("Words per document stats: mean {:.3f}, var {:.3f}, std {:.3f}".format(*stats))
        return ret_words, ret_pos, ret_voc


    # preprocess raw texts into word list
    def preprocess(self):
        if self.loaded_preprocessed:
            info("Skipping preprocessing, preprocessed data already loaded from {}.".format(self.serialization_path_preprocessed))
            return
        with tictoc("Preprocessing {}".format(self.name)):
            info("Mapping training set.")
            self.train, train_pos, self.vocabulary = self.preprocess_single_chunk(self.train, track_vocabulary=True)
            info("Mapping test set.")
            self.test, test_pos, _ = self.preprocess_single_chunk(self.test)
            # set pos
            self.pos_tags = [train_pos, test_pos]
            # fix word order and get word indexes
            self.vocabulary = list(self.vocabulary)
            for index, word in enumerate(self.vocabulary):
                self.word_to_index[word] = index
                self.vocabulary_index.append(index)
            # add another for the missing word
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
        nltk.download("reuters")
        # get ids
        categories = reuters.categories()
        self.num_labels = len(categories)
        self.train_label_names, self.test_label_names = [], []
        idx2label_train, idx2label_test = {}, {}

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
                    if cat_index not in idx2label_train:
                        idx2label_train[cat_index] = cat
                else:
                    self.test.append(content)
                    self.test_target.append(cat_index)
                    if cat_index not in idx2label_test:
                        idx2label_test[cat_index] = cat

        if len(idx2label_test) != len(idx2label_train):
            error("{} number of label train/test mismatch: {} / {}".format(self.name, len(idx2label_test), len(idx2label_test)))
        if idx2label_test != idx2label_train:
            error("{} index-label mismatch".format(self.name, idx2label_test, idx2label_test))
        self.train_label_names, self.test_label_names = [[idx2label_test[i] for i in idx2label_test]] * 2
        self.test_label_names = list(self.test_label_names)
        self.train_target = np.asarray(self.train_target, np.int32)
        self.test_target = np.asarray(self.test_target, np.int32)
        return self.get_all_raw()

    def handle_raw(self, raw_data):
        # serialize
        write_pickled(self.serialization_path, raw_data)
        self.loaded_raw = True
        pass

    # raw path setter
    def get_raw_path(self):
        # dataset is downloadable
        pass

#class MultilingMMS:
#    name = "multiling-mms"
#    def get_raw_path():
#        pass
#

class ManualDataset(Dataset):
    """ Class to import a dataset from a folder.

    Expected format in the yml config:
    name: path/to/dataset_name.json

    In the above path, define dataset json as:
    {
        data:
            train:
                 [
                     {
                        text: "this is the document text"
                        labels: [0,2,3]
                     },
                     ...
                 ],
            test: [...]
        num_labels: 10
        label_names: ['cat', 'dog', ...]
        language: english
    }
    """
    language = "english"

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = basename(config.dataset.name)
        Dataset.__init__(self)

    def get_all_raw(self):
        data = Dataset.get_all_raw(self)
        # data["language"] = self.language
        # data["multilabel"] = self.is_multilabel()
        return data

    # raw path getter
    def get_raw_path(self):
        return self.config.dataset.name

    def fetch_raw(self, raw_data_path):
        # no limited dataset
        if self.name != self.base_name:
            return None

        with open(raw_data_path) as f:
            raw_data = json.load(f)
        return raw_data

    def handle_raw(self, raw_data):
        max_num_instance_labels = 0
        self.num_labels = raw_data["num_labels"]
        self.language = raw_data["language"]
        data = raw_data["data"]

        self.train, self.train_target = [], []
        self.test, self.test_target = [], []

        unique_labels = {"train": set(), "test": set()}
        for obj in data["train"]:
            self.train.append(obj["text"])
            lbls = obj["labels"]
            self.train_target.append(lbls)
            unique_labels["train"].update(lbls)
            max_num_instance_labels = len(lbls) if len(lbls) > max_num_instance_labels else max_num_instance_labels
        for obj in data["test"]:
            self.test.append(obj["text"])
            self.test_target.append(obj["labels"])
            unique_labels["test"].update(obj["labels"])

        self.language = "english"
        if "label_names" in raw_data:
            self.train_label_names = raw_data["label_names"]["train"]
            self.test_label_names = raw_data["label_names"]["test"]
        else:
            self.train_label_names, self.test_label_names = \
                [list(map(str, sorted(unique_labels[tt]))) for tt in ["train", "test"]]
        if max_num_instance_labels > 1:
            self.multilabel = True
        # write serialized data
        write_pickled(self.serialization_path, self.get_all_raw())

    def handle_raw_serialized(self, deserialized_data):
        Dataset.handle_raw_serialized(self, deserialized_data)
        # self.language = deserialized_data["language"]
        # self.multilabel = deserialized_data["multilabel"]

    def handle_serialized(self, deserialized_data):
        self.handle_raw_serialized(self, deserialized_data)

    def handle_preprocessed(self, deserialized_data):
        Dataset.handle_preprocessed(self, deserialized_data)

    def get_name(self):
        # get only the filename
        return basename(self.name)

    def get_base_name(self):
        # get only the base filename
        return basename(self.base_name)

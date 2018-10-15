import logging
import pickle
import os
from helpers import Config
from utils import error, tic, toc
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords



class Dataset:
    name = ""
    limited_name = ""
    serialization_dir = "serialization/datasets"
    preprocessed = False
    train, test = None, None

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
        pass

    # data getter
    def get_data(self):
        return self.train, self.test

    def get_targets(self):
        return self.train_target, self.test_target

    def get_num_labels(self):
        return self.num_labels


    def apply_limit(self, config):
        if config.option("data_limit"):
            logger = logging.getLogger()
            value = config.option("data_limit")
            if self.train:
                self.train = self.train[:value]
                self.test = self.test[:value]
                self.train_target = self.train_target[:value]
                self.test_target = self.test_target[:value]
                logger.info("Limiting {} to {} items.".format(self.name, value))
            self.limited_name = self.name + "_limited_" + str(value)
            self.set_paths(self.limited_name)

    def preprocess(self):
        logger = logging.getLogger()
        stopw = set(stopwords.words('english'))
        if self.preprocessed:
            logger.info("Skipping preprocessing, loading existing data from {}.".format(self.serialization_path_preprocessed))
            return
        tic()
        filter = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
        logger.info("Preprocessing {}".format(self.name))
        logger.info("Mapping training set to word sequence.")
        for i in range(len(self.train)):
            processed = text_to_word_sequence(self.train[i], filters=filter, lower=True, split=' ')
            processed = [p for p in processed if p not in stopw]
            self.train[i] = processed
        logger.info("Mapping test set to word sequence.")
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

    def make(self, config):
        Dataset.make(self, config)
        logger = logging.getLogger()

        self.apply_limit(config)
        # if preprocessed data already exists, load them
        if os.path.exists(self.serialization_path_preprocessed):
            with open(self.serialization_path_preprocessed, "rb") as f:
                self.train, self.train_target, self.train_label_names, self.test, self.test_target, self.test_label_names = pickle.load(f)
                self.num_labels = len(self.train_label_names)
                self.preprocessed = True
                return
        self.suspend_limit()

        # else, check if the downloaded & serialized raw data exists
        if os.path.exists(self.serialization_path) and os.path.isfile(self.serialization_path):
            logger.info("Loading {} from serialization path {}".format(self.name, self.serialization_path))
            with open(self.serialization_path, "rb") as f:
                deser = pickle.load(f)
                train, test = deser[0], deser[1]
        else:
            # else, fetch from scikit-learn
            logger.info("Downloading {} via sklearn".format(self.name))
            seed = Config().get_seed()
            train = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
            test = fetch_20newsgroups(subset='train', shuffle=True, random_state=seed)
            # write
            logger.info("Writing {} dataset from serialization path {}".format(self.name, self.serialization_path))
            with open(self.serialization_path, "wb") as f:
                pickle.dump([train, test], f)

        # results are sklearn bunches
        logger.info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        # map to train/test/categories
        self.train, self.test = train.data, test.data
        self.train_target, self.test_target = train.target, test.target
        self.train_label_names = train.target_names
        self.test_label_names = test.target_names
        self.num_labels = len(self.train_label_names)

        self.apply_limit(config)


import logging
import pickle
import os
from helpers import error, Config, tic, toc
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import text_to_word_sequence



class Dataset:
    name = ""
    serialization_dir = "datasets"

    def __init__(self, name):
        self.name = name
        pass

    def make(self, config):
        if config.option("data_limit"):
            logger = logging.getLogger()
            value = config.option("data_limit")
            self.train = self.train[:value]
            self.test = self.test[:value]
            logger.info("Limiting {} to {} items.".format(self.name, value))

    # data getter
    def get_data(self):
        return self.train, self.test
    def get_targets(self):
        return self.train_target, self.test_target


    def preprocess(self):
        logger = logging.getLogger()
        tic()
        filter = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
        logger.info("Preprocessing {}".format(self.name))
        for i in range(len(self.train)):
            processed = text_to_word_sequence(self.train[i], filters=filter, lower=True, split=' ')
            self.train[i] = processed
            # print(processed)
        for i in range(len(self.test)):
            processed = text_to_word_sequence(self.test[i], filters=filter, lower=True, split=' ')
            self.test[i] = processed
            # print(processed)
        toc("Preprocessing")


    def get_name(self):
        return self.name


class TwentyNewsGroups(Dataset):
    name = "20newsgroups"

    def __init__(self):
        Dataset.__init__(self, TwentyNewsGroups.name)
        self.serialization_path = "{}/{}.pickle".format(self.serialization_dir, self.name)

    def make(self, config):
        logger = logging.getLogger()
        # if already downloaded and serialized, load it
        if os.path.exists(self.serialization_path) and os.path.isfile(self.serialization_path):
            logger.info("Loading {} from serialization path {}".format(self.name, self.serialization_path))
            with open(self.serialization_path, "rb") as f:
                deser = pickle.load(f)
                train, test = deser[0], deser[1]
        else:
            logger.info("Downloading {} via sklearn".format(self.name))
            # fetch from scikit-learn
            # alternatively you can se sklearn manually with
            # sklearn.datasets.load_files(data_folder) to load each category data
            seed = Config().get_seed()
            train = fetch_20newsgroups(subset='train', shuffle=False, random_state=seed)
            test = fetch_20newsgroups(subset='train', shuffle=False, random_state=seed)
            # write
            logger.info("Writing to {} from serialization path {}".format(self.name, self.serialization_path))
            if not os.path.exists(self.serialization_dir):
                os.mkdir(self.serialization_dir)
            with open(self.serialization_path, "wb") as f:
                pickle.dump([train, test], f)
        # results are sklearn bunches
        logger.info("Got {} and {} train / test samples".format(len(train.data), len(test.data)))
        # map to train/test/categories
        self.train, self.test = train.data, test.data
        self.train_target, self.test_target = train.target, test.target

        Dataset.make(self, config)


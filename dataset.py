import logging
import pickle
import os
from helpers import error, Config
from sklearn.datasets import fetch_20newsgroups
# from keras.preprocessing.text import text_to_word_sequence



class Dataset:
    name = ""
    serialization_dir = "datasets"

    def __init__(self, name):
        self.name = name
        pass

    def make(self):
        logging.getLogger().warning("Attempted to make an abstract dataset!")
        pass

    def preprocess(self):
        for i in range(len(self.train)):
            pass
#           processed = text_to_word_sequence(self.train[i], filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
#            self.train[i] = processed




class TwentyNewsGroups(Dataset):
    name = "20newsgroups"

    def __init__(self):
        Dataset.__init__(self, TwentyNewsGroups.name)
        self.serialization_path = "{}/{}.pickle".format(self.serialization_dir, self.name)

    def make(self):
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

    # data getter
    def get_data(self):
        return self.train, self.test

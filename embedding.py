import os
import logging
import pandas
import csv
import pickle
from helpers import error

class Embedding():
    name = ""
    serialization_dir = "embeddings"

class Glove(Embedding):
    name = "glove"
    def __init__(self, params):
        dim = params[0]
        raw_data_path = os.path.join(self.serialization_dir, "glove.6B.{}d.txt".format(dim))
        pickled_path = os.path.join(self.serialization_dir, "glove6B.{}d.pickle".format(dim))
        logger = logging.getLogger()
        # read pickled existing data
        if os.path.exists(pickled_path):
            with open(pickled_path, "rb") as f:
                self.embeddings = pickle.load(f)
                return
        # load existing raw data
        if os.path.exists(raw_data_path):
            logger.info("Reading raw embedding data from {}".format(raw_data_path))
            self.words = pandas.read_csv(raw_data_path, index_col=0, header=None, sep=" ", quoting=csv.QUOTE_NONE).values
            return
        # gotta download the raw data
        error("Downloaded glove embeddings missing. Get them from https://nlp.stanford.edu/projects/glove/")

    # transform input texts to embeddings
    def map_text(self, texts):
        pass



class Word2vec(Embedding):
    name = "word2vec"
    def __init__(self):
        pass

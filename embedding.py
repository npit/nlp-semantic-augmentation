import os
import logging
import pandas as pd
import csv
import pickle
from helpers import error, tic, toc
import numpy as np


class Embedding():
    name = ""
    serialization_dir = "embeddings"
    words = None
    dataset_embeddings = None
    embeddings = None
    words_to_numeric_idx  = None

    def read_pickled(self, pickled_path):
        logger = logging.getLogger()
        logger.info("Reading pickled embedding data from {}".format(pickled_path))
        with open(pickled_path, "rb") as f:
            data =  pickle.load(f)
            self.embeddings, self.words_to_numeric_idx = data[0], data[1]

    def write_pickled(self, pickled_path):
        tic()
        logging.getLogger().info("Pickling embedding data to {}".format(pickled_path))
        with open(pickled_path, "wb") as f:
            # pickle.dump([self.words, self.embeddings], f)
            pickle.dump([self.embeddings, self.words_to_numeric_idx], f)
        toc("Pickling")

    def get_words(self):
        if self.words is None:
            self.words = []
            for dset in self.dataset_embeddings:
                self.words.append([])
                for document in dset:
                    doc_words = document.index.tolist()
                    self.words[-1].append(doc_words)
        return self.words


class Glove(Embedding):
    name = "glove"

    def __init__(self, params):
        dim = params[0]
        raw_data_path = os.path.join(self.serialization_dir, "glove.6B.{}d.txt".format(dim))
        pickled_path = os.path.join(self.serialization_dir, "glove6B.{}d.pickle".format(dim))
        logger = logging.getLogger()

        # read pickled existing data
        if os.path.exists(pickled_path):
            self.read_pickled(pickled_path)
            return

        # load existing raw data
        if os.path.exists(raw_data_path):
            logger.info("Reading raw embedding data from {}".format(raw_data_path))
            tic()
            self.embeddings = pd.read_csv(raw_data_path, index_col = 0, header=None, sep=" ", quoting=csv.QUOTE_NONE)
            self.words_to_numeric_idx = {word:i for word,i in zip(self.embeddings.index.values.tolist(), range(len(self.embeddings.index))) }
            toc("Reading raw data")
            # pickle them
            self.write_pickled(pickled_path)
            return

        # else, gotta download the raw data
        error("Downloaded glove embeddings missing. Get them from https://nlp.stanford.edu/projects/glove/")

    # transform input texts to embeddings
    def map_text(self, dset):
        logger = logging.getLogger()
        logger.info("Mapping {} to {} embeddings.".format(dset.name, self.name))
        text_bundles = dset.train, dset.test
        self.dataset_embeddings = []
        # loop over input text bundles (e.g. train & test)
        for i in range(len(text_bundles)):
            self.dataset_embeddings.append([])
            tic()
            logger.info("Mapping text bundle {}/{}: {} texts".format(i+1, len(text_bundles), len(text_bundles[i])))
            hist = {w: 0 for w in self.words_to_numeric_idx}
            hist_missing = {}
            for j in range(len(text_bundles[i])):
                word_list = text_bundles[i][j]
                logger.debug("Text {}/{}".format(j+1, len(text_bundles[i])))
                text_embeddings = self.embeddings.loc[word_list]

                # stats
                missing_words = text_embeddings[text_embeddings.isnull().any(axis=1)].index.tolist()
                text_embeddings = self.embeddings.loc[word_list].dropna()
                present_words = text_embeddings.index.tolist()
                for w in present_words:
                    hist[w] += 1
                for m in missing_words:
                    if m not in hist_missing:
                        hist_missing[m] = 0
                    hist_missing[m] += 1

                self.dataset_embeddings[-1].append(text_embeddings)

            toc("Embedding mapping for text bundle {}/{}".format(i+1, len(text_bundles)))

            num_words_hit, num_hit = sum([1 for v in hist if hist[v] > 0]), sum(hist.values())
            num_words_miss, num_miss = len(hist_missing.keys()), sum(hist_missing.values())
            num_total = sum(list(hist.values()) + list(hist_missing.values()))

            logger.info("Found {} instances or {:.3f} % of total {}, for {} words.".format(num_hit, num_hit/num_total*100, num_total, num_words_hit))
            logger.info("Missed {} instances or {:.3f} % of total {}, for {} words.".format(num_miss, num_miss/num_total*100, num_total, num_words_miss))
            # import pdb; pdb.set_trace()

            # log missing words
            missing_filename = os.path.join(self.serialization_dir, self.name + "_" + dset.name + ".txt")
            logger.info("Writing missing words to {}".format(missing_filename))
            with open(missing_filename, "w") as f:
                f.write("\n".join(hist_missing.keys()))
            


class Word2vec(Embedding):
    name = "word2vec"
    def __init__(self):
        pass

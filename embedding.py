import os
import logging
import pandas as pd
import csv
import pickle
from utils import error, tic, toc
import numpy as np


class Embedding():
    name = ""
    serialization_dir = "serializations/embeddings"
    words = []
    dataset_embeddings = None
    embeddings = None
    words_to_numeric_idx  = None
    missing = []

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
        return self.words

    def get_data(self):
        return self.dataset_embeddings

    # prepare embedding data to be ready for classification
    def prepare(self, config):
        # save the words
        for dset in self.dataset_embeddings:
            self.words.append([])
            for document in dset:
                doc_words = document.index.tolist()
                self.words[-1].append(doc_words)

        aggregation = config.get_aggregation()
        if aggregation == "avg":
            # average all word vectors in the doc
            for dset_idx in range(len(self.dataset_embeddings)):
                aggregated_doc_vectors = []
                for doc_dict in self.dataset_embeddings[dset_idx]:
                    aggregated_doc_vectors.append(np.mean(doc_dict.values, axis=0))
                self.dataset_embeddings[dset_idx] = np.concatenate(aggregated_doc_vectors).reshape(len(aggregated_doc_vectors), self.embedding_dim)





class Glove(Embedding):
    name = "glove"

    def __init__(self, params):
        self.embedding_dim = int(params[0])
        if not os.path.exists(self.serialization_dir):
            os.makedirs(self.serialization_dir, exist_ok=True)
        raw_data_path = os.path.join("embeddings/glove.6B.{}d.txt".format(self.embedding_dim))
        pickled_path = os.path.join(self.serialization_dir, "glove6B.{}d.pickle".format(self.embedding_dim))
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
        error("Downloaded glove embeddings missing from {}. Get them from https://nlp.stanford.edu/projects/glove/".format(raw_data_path))


    # transform input texts to embeddings
    def map_text(self, dset, config):
        logger = logging.getLogger()
        if config.option("data_limit"):
            serialization_path = os.path.join(self.serialization_dir, "embeddings_mapped_{}_limit{}_{}.pickle".format(dset.name, config.option("data_limit"), self.name))
        else:
            serialization_path = os.path.join(self.serialization_dir, "embeddings_mapped_{}_{}.pickle".format(dset.name, self.name))
        if os.path.exists(serialization_path):
            with open(serialization_path, "rb") as f:
                [self.dataset_embeddings, self.missing_words] = pickle.load(f)
            return

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

            self.missing.append(hist_missing)

        # write
        with open(serialization_path, "wb") as f:
            pickle.dump([self.dataset_embeddings, self.missing], f)
        # log missing words
        for d in range(len(self.missing)):
            l = ['train', 'text']
            missing_filename = os.path.join(self.serialization_dir, "missing_words_" + self.name + "_" + dset.name + l[d] + ".txt")
            logger.info("Writing missing words to {}".format(missing_filename))
            with open(missing_filename, "w") as f:
                f.write("\n".join(self.missing[d].keys()))


class Word2vec(Embedding):
    name = "word2vec"
    def __init__(self):
        pass

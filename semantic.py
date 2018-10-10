# import gensim
import os
import copy
import pickle
import logging
from helpers import tic, toc
from nltk.corpus import wordnet as wn

class Wordnet:
    name="wordnet"
    serialization_dir = "semantic_data"
    word_synset_cache = {}

    synset_freqs = []
    synset_tfidf_freqs = []
    dataset_freqs = []
    dataset_minmax_freqs = []
    assignments = {}

    # merge list of document-wise frequency dicts
    # to a single, dataset-wise frequency dict
    def doc_to_dset_freqs(self, freq_dict_list):
        dataset_freqs = {}
        for doc_dict in freq_dict_list:
            for synset, freq in doc_dict.items():
                if synset not in dataset_freqs:
                    dataset_freqs[synset] = 0
                dataset_freqs[synset] += freq
        return dataset_freqs


    # map a dataset
    def map_dset(self, dset, dataset_name):
        logger = logging.getLogger()
        current_synset_freqs = []
        tic()
        for wl, word_list in enumerate(dset):
            doc_freqs = {}
            for w, word in enumerate(word_list):
                logger.info("Semantic processing for word {}/{} ({}) of doc {}/{}".format(w+1, len(word_list), word, wl+1, len(dset)))
                synset, doc_freqs = self.get_synset(word, doc_freqs)
                if not synset: continue
                self.process_synset(synset)
            current_synset_freqs.append(doc_freqs)
        toc("Document-level mapping and frequency computation")
        self.synset_freqs.append(current_synset_freqs)
        # merge to dataset-wise synset frequencies
        tic()
        dataset_freqs = self.doc_to_dset_freqs(current_synset_freqs)
        self.dataset_freqs.append(dataset_freqs)
        toc("Dataset-level frequency computation")
        # compute tf-idf
        tic()
        tfidf_freqs = []
        for doc_dict in range(len(current_synset_freqs)):
            ddict = {}
            for synset in current_synset_freqs[doc_dict]:
                ddict[synset] =  current_synset_freqs[doc_dict][synset] / dataset_freqs[synset]
            tfidf_freqs.append(ddict)
        self.synset_tfidf_freqs.append(tfidf_freqs)
        toc("tf-idf computation")



    # function to map words to wordnet concepts
    def map_text(self, datasets_words, dataset_name):

        logger = logging.getLogger()
        serialization_path = os.path.join(self.serialization_dir, "{}_{}_assignments".format(dataset_name, self.name))
        if os.path.exists(serialization_path):
            logger.info("Loading existing mapped semantic information.")
            with open(serialization_path, "rb") as f:
                self.assignments, self.synset_freqs, self.dataset_freqs, self.synset_tfidf_freqs = pickle.load(f)
            return

        tic()
        self.synset_freqs = [[] for _ in datasets_words]
        for d, dset in enumerate(datasets_words):
            self.map_dset(dset, dataset_name + "{}".format(d+1))

        # write results: word assignments, raw, dataset-wise and tf-idf weights
        logger.info("Writing semantic assignment results.")
        if not os.path.exists(self.serialization_dir):
            os.mkdir(self.serialization_dir)
        # import pdb; pdb.set_trace()
        with open(serialization_path, "wb") as f:
            pickle.dump([self.assignments, self.synset_freqs, self.dataset_freqs, self.synset_tfidf_freqs], f)

        logger.info("Semantic mapping completed.")



    # function to get a synset from a word, using the wordnet api
    # and a local word cache. Updates synset frequencies as well.
    def get_synset(self, word, freqs):
        if word in self.word_synset_cache:
            # print("Cache hit:", self.word_synset_cache)
            # print("freqs:", freqs)
            synset = self.word_synset_cache[word]
            if synset not in freqs:
                freqs[synset] = 0
        else:
            # print("Cache miss:", self.word_synset_cache)
            # print("freqs:", freqs)
            synsets = wn.synsets(word)
            if not synsets:
                return None, freqs
            synset = synsets[0]._name
            freqs[synset] = 0
            self.word_synset_cache[word] = synset

        freqs[synset] += 1
        self.assignments[word] = synset
        return synset, freqs

    # function that applies the required processing
    # once a synset has been found in the input text
    def process_synset(self, synset):
        # spreading activation
        pass


    # function that applies post-processing for a collected synset graph 
    def postprocess_synset(self, synset):
        # frequency / tf-idf filtering - do that with synset freqs
        # ingoing / outgoing graph ratio
        pass

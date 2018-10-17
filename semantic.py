# import gensim
import os
import copy
import pickle
import logging
from utils import tic, toc, error, info, debug
from nltk.corpus import wordnet as wn

class Wordnet:
    name="wordnet"
    serialization_dir = "serializations/semantic_data"
    word_synset_cache = {}

    synset_freqs = []
    synset_tfidf_freqs = []
    dataset_freqs = []
    dataset_minmax_freqs = []
    assignments = {}



    def apply_freq_filtering(self, freq_dict_list, dataset_freqs, force_reference=False):
        info("Applying synset frequency filtering with a threshold of {}".format(self.semantic_freq_threshold))
        tic()
        # delete from dataset-level dicts
        synsets_to_delete = set()
        for synset in dataset_freqs:
            if dataset_freqs[synset] < self.semantic_freq_threshold:
                synsets_to_delete.add(synset)
        toc("Dataset-level frequency filtering")
        # if forcing reference, we can override the freq threshold for these synsets
        if force_reference:
            orig_num = len(synsets_to_delete)
            synsets_to_delete = [x for x in synsets_to_delete if x not in self.reference_synsets]
            info("Limiting synsets-to-delete from {} to {} due to forcing to reference synset set".format(orig_num, len(synsets_to_delete)))
        if not synsets_to_delete:
            return  freq_dict_list, dataset_freqs
        info("Will remove {}/{} synsets due to a freq threshold of {}".format(len(synsets_to_delete), len(dataset_freqs), self.semantic_freq_threshold))
        tic()
        # delete
        for synset in synsets_to_delete:
            del dataset_freqs[synset]
            for doc_dict in freq_dict_list:
                if synset in doc_dict:
                    del doc_dict[synset]
        toc("Document-level frequency filtering")
        return  freq_dict_list, dataset_freqs

    def make(self, config):
        self.serialization_dir = os.path.join(config.get_serialization_dir(), "semantic_data")
        self.semantic_freq_threshold = config.get_semantic_freq_threshold()

    # merge list of document-wise frequency dicts
    # to a single, dataset-wise frequency dict
    def doc_to_dset_freqs(self, freq_dict_list, force_reference = False):
        dataset_freqs = {}
        for doc_dict in freq_dict_list:
            for synset, freq in doc_dict.items():
                if synset not in dataset_freqs:
                    dataset_freqs[synset] = 0
                dataset_freqs[synset] += freq
        # frequency filtering, if defined
        if self.semantic_freq_threshold:
            freq_dict_list, dataset_freqs = self.apply_freq_filtering(freq_dict_list, dataset_freqs, force_reference)

        # complete document-level freqs with zeros for dataset-level synsets missing in the document level
        #for d, doc_dict in enumerate(freq_dict_list):
        #    for synset in [s for s in dataset_freqs if s not in doc_dict]:
        #        freq_dict_list[d][synset] = 0
        return dataset_freqs, freq_dict_list


    # map a dataset
    def map_dset(self, dset, dataset_name, store_reference_synsets = False, force_reference_synsets = False):
        logger = logging.getLogger()
        if force_reference_synsets:
            # restrict discoverable synsets to a predefined selection
            # used for the test set where not encountered synsets are unusable
            logger.info("Restricting synsets to the reference synset set of {} entries.".format(len(self.reference_synsets)))
            #import pdb;pdb.set_trace();

        current_synset_freqs = []
        tic()
        for wl, word_list in enumerate(dset):
            debug("Semantic processing for document {}/{}".format(wl+1, len(dset)))
            doc_freqs = {}
            for w, word in enumerate(word_list):
                synset, doc_freqs = self.get_synset(word, doc_freqs, force_reference_synsets)
                if not synset: continue
                self.process_synset(synset)
            current_synset_freqs.append(doc_freqs)
        toc("Document-level mapping and frequency computation")
        # merge to dataset-wise synset frequencies
        tic()
        dataset_freqs, current_synset_freqs = self.doc_to_dset_freqs(current_synset_freqs, force_reference = force_reference_synsets)
        self.dataset_freqs.append(dataset_freqs)
        self.synset_freqs.append(current_synset_freqs)
        toc("Dataset-level frequency computation")

        if store_reference_synsets:
            self.reference_synsets = set((dataset_freqs.keys()))
        # compute tf-idf
        tic()
        tfidf_freqs = []
        for doc_dict in range(len(current_synset_freqs)):
            ddict = {}
            for synset in current_synset_freqs[doc_dict]:
                if dataset_freqs[synset] > 0:
                    ddict[synset] = current_synset_freqs[doc_dict][synset] / dataset_freqs[synset]
                else:
                    ddict[synset] = 0

                tfidf_freqs.append(ddict)
        self.synset_tfidf_freqs.append(tfidf_freqs)
        toc("tf-idf computation")


    def disambiguate_synsets(self, synsets):
        return synsets[0]._name

    def disambiguate(self, config):
        disam = config.get_disambiguation()
        if disam == 'POS':
            # part-of-speech filtering
            pass
        elif disam == 'embedding-centroid':
            # generate closest synset embeddings
            # assign to closest embedding
            pass
        elif disam == "prior":
            # select the synset with the highest prior prob
            pass
        else:
            error("Undefined disambiguation method: " + disam)


    # function to map words to wordnet concepts
    def map_text(self, datasets_words, dataset_name):
        logger = logging.getLogger()
        serialization_path = os.path.join(self.serialization_dir, "{}_{}_assignments".format(dataset_name, self.name))
        # check if data is already extracted & serialized
        if os.path.exists(self.serialization_dir):
            os.makedirs(self.serialization_dir, exist_ok=True)
        if os.path.exists(serialization_path):
            info("Loading existing mapped semantic information.")
            tic()
            with open(serialization_path, "rb") as f:
                # load'em.
                # assignments: word -> synset assignment
                # synset_freqs: frequencies of synsets per document
                # dataset_freqs: frequencies of synsets aggregated per dataset
                # synset_tfidf: tf-idf frequencies of synsets per document (df is per dataset)
                self.assignments, self.synset_freqs, self.dataset_freqs, self.synset_tfidf_freqs = pickle.load(f)
                toc("Loading")
            return

        # process the data
        self.synset_freqs = []
        for d, dset in enumerate(datasets_words):
            info("Extracting semantic information from dataset {}/{}".format(d+1, len(datasets_words)))
            # process data within a dataset portion
            self.map_dset(dset, dataset_name + "{}".format(d+1),
                          store_reference_synsets=d == 0,
                          force_reference_synsets=d > 0)

        # write results: word assignments, raw, dataset-wise and tf-idf weights
        logger.info("Writing semantic assignment results.")
        if not os.path.exists(self.serialization_dir):
            os.mkdir(self.serialization_dir)
        # import pdb; pdb.set_trace()
        with open(serialization_path, "wb") as f:
            pickle.dump([self.assignments, self.synset_freqs,
                         self.dataset_freqs, self.synset_tfidf_freqs], f)

        info("Semantic mapping completed.")



    # function to get a synset from a word, using the wordnet api
    # and a local word cache. Updates synset frequencies as well.
    def get_synset(self, word, freqs, force_reference_synsets = False):
        if word in self.word_synset_cache:
            # print("Cache hit:", self.word_synset_cache)
            # print("freqs:", freqs)
            synset = self.word_synset_cache[word]

            if force_reference_synsets:
                if synset not in self.reference_synsets:
                    return None, freqs

            if synset not in freqs:
                freqs[synset] = 0
        else:
            # print("Cache miss:", self.word_synset_cache)
            # print("freqs:", freqs)
            synsets = wn.synsets(word)
            if not synsets:
                return None, freqs
            synset = self.disambiguate_synsets(synsets)
            if force_reference_synsets:
                if synset not in self.reference_synsets:
                    return None, freqs
            freqs[synset] = 0
            self.word_synset_cache[word] = synset

        freqs[synset] += 1
        if word not in self.assignments:
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

    # get requested information to use
    def get_data(self, config):
        # map dicts to vectors
        synset_order = sorted(self.dataset_freqs[0].keys())
        for d, dset in enumerate(self.dataset_freqs):
            #import pdb; pdb.set_trace()
            if not set(synset_order) == set(dset.keys()):
                print(len(synset_order))
                print(len(dset.keys()))
                error("synset mismatch in dataset and synset order from first")
        semantic_document_vectors = [[] for _ in range(len(self.synset_freqs)) ]

        semtype = config.get_semantic_type()
        if semtype  == "freq":
            # get raw frequencies
            for d in range(len(self.synset_freqs)):
                for doc_dict in self.synset_freqs[d]:
                    doc_vector = [doc_dict[s] if s in doc_dict else 0 for s in synset_order]
                    semantic_document_vectors[d].append(doc_vector)
        else:
            error("Unimplemented semantic vector method: {}.".format(semtype))

        return semantic_document_vectors

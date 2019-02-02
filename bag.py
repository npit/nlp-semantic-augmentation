from utils import error, tictoc, debug, info
import defs
import tqdm
import numpy as np


class Bag:
    term_list = None
    global_weights = None
    weights = None
    term_weighting_func = None
    term_greenlight_func = None
    use_fixed_term_list = None
    do_filter = None
    filter_type = None
    filter_quantity = None
    term_delineation_func = None

    @staticmethod
    def delineate_words(collection):
        return [word_info for word_info in collection]

    @staticmethod
    def extract_word(word_info):
        return word_info[0]

    def __init__(self):
        self.term_list = []
        self.global_weights = {}
        self.use_fixed_term_list = False
        # non-fixed term list as the default
        self.term_weighting_func = self.unit_term_weighting
        self.term_greenlight_func = lambda x: True
        # words from a text, as default term delineation
        self.term_delineation_func = Bag.delineate_words
        # extract the useful parts from a term -- default is word from word info tuple
        self.term_extraction_func = Bag.extract_word

        self.filter_type = None
        self.filter_quantity = None
        self.do_filter = False

    # term filtering
    # #####################
    # filter the terms
    def filter_terms(self):
        if self.filter_type is None:
            return
        info("Filtering terms with type / threshold: {}".format((self.filter_type, self.filter_quantity)))
        if self.filter_type == defs.limit.frequency:
            # discard terms with less frequency than the threshold.
            # make a single pass to mark the terms to keep
            terms_to_retain = set()
            for doc_dict in self.weights:
                for term, freq in doc_dict.items():
                    if freq >= self.filter_quantity:
                        terms_to_retain.add(term)
            self.delete_terms(terms_to_retain)

        elif self.filter_type == defs.limit.top:
            if len(self.term_list) == self.filter_quantity:
                return
            # sort w.r.t. collection-wise frequency, keep <threshold> first
            sorted_global_freqs = sorted(self.global_weights, reverse=True, key=lambda tok: self.global_weights[tok])
            terms_to_retain = sorted_global_freqs[:self.filter_quantity]
            self.delete_terms(terms_to_retain)
        else:
            error("Undefined limiting type: {}".format(self.filter_type))

    # remove terms from the accumulated collections
    def delete_terms(self, terms_to_retain):
        # do not delete terms if handling a fixed list
        for i, doc_dict in enumerate(self.weights):
            self.weights[i] = {tok: doc_dict[tok] for tok in doc_dict if tok in terms_to_retain}
        self.global_weights = {term: self.global_weights[term] for term in terms_to_retain}
        # limit the term list itself
        self.term_list = [t for t in self.term_list if t in terms_to_retain]

    # setters
    # #####################
    # enable term filtering
    def set_term_filtering(self, filter_type, filter_quantity):
        self.do_filter = True
        self.filter_type = filter_type
        self.filter_quantity = filter_quantity

    # set the term list of the bag
    def set_term_list(self, term_list):
        self.use_fixed_term_list = True
        self.term_list = list(term_list)
        self.global_weights = {key: 0 for key in self.term_list}
        self.num_terms = len(self.term_list)
        self.term_weighting_func = self.unit_term_weighting
        self.term_greenlight_func = self.check_term_in_list

    # override term extraction function
    def set_term_extraction_function(self, func):
        self.term_extraction_func = func

    # override term delineation function
    def set_term_delineation_function(self, func):
        self.term_delineation_func = func

    # override default term processing function
    def set_term_weighting_function(self, func):
        self.term_weighting_func = func

    # override default term checking function
    def set_term_greenlight_function(self, func):
        self.term_greenlight_func = func

    def populate_all_data(self, weights, global_weights, term_list):
        self.global_weights = global_weights
        self.term_list = term_list
        self.weights = weights

    # frequency updater function
    def update_term_frequency(self, item, weight, weights_dict):
        # accumulate DF
        if item not in self.global_weights:
            self.global_weights[item] = weight
        else:
            self.global_weights[item] += weight
        # accumulate TF
        if item not in weights_dict:
            weights_dict[item] = weight
        else:
            weights_dict[item] += weight
        return weights_dict

    # checks wether the term is valid to be included in the bag
    def check_term_in_list(self, term):
        # if a term list has been supplied, only use terms in it
        return term in self.term_list

    # processes an term of an instance, producing pairs of vector_index and frequency_count
    # Applied for a non-fixed term list, returning the term itself
    def unit_term_weighting(self, term):
        if self.term_greenlight_func(term):
            return {term: 1}
        return {}

    def map_collection(self, text_collection):
        # present word tracking only available for the default (word-wise) delineation
        do_track_present_words = False if self.term_delineation_func != Bag.delineate_words else True

        # collection-wise information
        self.weights, self.present_terms, self.present_term_indexes = [], [], []

        # global term-wise frequencies
        with tqdm.tqdm(desc="Creating bow vectors", total=len(text_collection), ascii=True, ncols=100, unit="collection") as pbar:
            for t, word_info_list in enumerate(text_collection):
                pbar.set_description("Text {}/{}".format(t + 1, len(text_collection)))
                pbar.update()
                text_term_freqs = {}
                present_words = []
                # delineate and extract terms of interest
                term_list = [self.term_extraction_func(x) for x in self.term_delineation_func(word_info_list)]
                # iterate
                for term in term_list:
                    # process term, and produce weight information of the processed result
                    term_weights = self.term_weighting_func(term)
                    # if the word produced weights, add it to the present words
                    if not term_weights:
                        continue
                    # add the term as a valid / present one
                    if do_track_present_words:
                        present_words.append(term)
                    # accumulate the term-weight results
                    for item, weight in term_weights.items():
                        # use item index as the key
                        text_term_freqs = self.update_term_frequency(item, weight, text_term_freqs)

                # completed document parsing
                # add required results
                self.weights.append(text_term_freqs)
                if do_track_present_words:
                    self.present_term_indexes.append([term_list.index(p) for p in present_words])
                    self.present_terms.append(len(present_words))
            # completed collection parsing - wrap up
            if not self.use_fixed_term_list:
                # if we computed the term list, store it
                self.term_list = list(self.global_weights.keys())

        # apply filtering, if selected
        self.filter_terms()
        info("Mapped the collection to frequencies of {} terms.".format(len(self.term_list)))

    # getters
    # #####################
    def get_weights(self):
        return self.weights

    def get_global_weights(self):
        return self.global_weights

    def get_present_term_indexes(self):
        return self.present_term_indexes

    def get_present_terms(self):
        return self.present_terms

    def get_term_list(self):
        return self.term_list

    @staticmethod
    def generate_dense(weights, term_list):
        for dset_idx in range(len(weights)):
            container = np.zeros((0, len(term_list)), np.float32)
            for doc_idx, doc_dict in enumerate(weights[dset_idx]):
                container = np.append(container, np.asarray([[doc_dict[s] if s in doc_dict else 0 for s in term_list]], np.float32), axis=0)
            weights[dset_idx] = container
        return weights


class TFIDF(Bag):

    def __init__(self):
        Bag.__init__(self)

    def map_collection(self, text_collection):
        Bag.map_collection(self, text_collection)
        # apply IDF
        self.idf_normalize()

    def idf_normalize(self, input_data=None):
        with tictoc("TFIDF normalization"):
            if input_data is not None:
                # just process the input
                self.weights, self.global_weights = input_data
            # normalize
            for vector_idx in range(len(self.weights)):
                for term_key in self.weights[vector_idx]:
                    self.weights[vector_idx][term_key] = \
                        self.weights[vector_idx][term_key] / self.global_weights[term_key]

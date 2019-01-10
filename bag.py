import numpy as np
from utils import error, tictoc, debug, info
import defs
from copy import copy


class Bag:
    token_list = None
    global_freqs = None
    output_vectors = None
    element_processing_func = None
    element_checking_func = None
    use_fixed_token_list = None
    filter_type = None
    filter_quantity = None
    calculate_global = None

    def __init__(self):
        self.token_list = []
        self.global_freqs = {}
        self.use_fixed_token_list = False
        # non-fized token list as the default
        self.element_processing_func = self.process_element
        self.element_checking_func = lambda x: True
        self.filter_type = None
        self.filter_quantity = None
        self.calculate_global = False

    # enable token filtering
    def set_token_filtering(self, filter_type, filter_quantity):
        self.filter_type = filter_type
        self.filter_quantity = filter_quantity

    # filter the tokens
    def filter_tokens(self):
        if self.filter_type is None:
            return
        info("Filtering tokens with type / threshold: {}".format((self.filter_type, self.filter_quantity)))
        if self.filter_type == defs.limit.frequency:
            # discard tokens with less frequency than the threshold.
            # make a single pass to mark the tokens to keep
            tokens_to_retain = set()
            for doc_dict in self.output_vectors:
                for token, freq in doc_dict.items():
                    if freq >= self.filter_quantity:
                        tokens_to_retain.add(token)
            self.delete_tokens(tokens_to_retain)

        elif self.filter_type == defs.limit.top:
            # sort w.r.t. collection-wise frequency, keep <threshold> first
            sorted_global_freqs = sorted(self.global_freqs, reverse=True, key=lambda tok: self.global_freqs[tok])
            tokens_to_retain = sorted_global_freqs[:self.filter_quantity]
            self.delete_tokens(tokens_to_retain)
        else:
            error("Undefined limiting type: {}".format(self.filter_type))

    # remove tokens from the accumulated collections
    def delete_tokens(self, tokens_to_retain):
        # do not delete tokens if handling a fixed list
        for i, doc_dict in enumerate(self.output_vectors):
            self.output_vectors[i] = {tok: doc_dict[tok] for tok in doc_dict if tok in tokens_to_retain}
        if self.calculate_global:
            self.global_freqs = {token: self.global_freqs[token] for token in tokens_to_retain}

    # set the token list of the bag
    def set_token_list(self, token_list):
        self.use_fixed_token_list = True
        self.token_list = list(token_list)
        self.global_freqs = {key: 0 for key in self.token_list}
        self.num_tokens = len(self.token_list)
        self.element_checking_func = self.process_element
        self.element_processing_func = self.check_element_in_list

    def get_token_list(self):
        return self.token_list

    # frequency updater function
    def update_token_frequency(self, item, weight, weights_dict):
        # do collection-wise frequencies, if specified
        if self.calculate_global:
            # accumulate DF
            if item not in self.global_freqs:
                self.global_freqs[item] = weight
            else:
                self.global_freqs[item] += weight
        if item not in weights_dict:
            weights_dict[item] = weight
        else:
            weights_dict[item] += weight
        return weights_dict

    # override default element processing function
    def set_element_processing_function(self, func):
        self.element_processing_func = func

    # override default element checking function
    def set_element_checking_function(self, func):
        self.element_checking_func = func

    # checks wether the element is valid to be included in the bag
    def check_element_in_list(self, element):
        # if a token list has been supplied, only use tokens in it
        return element in self.token_list

    # processes an element of an instance, producing pairs of vector_index and frequency_count
    # Applied for a non-fixed token list, returning the token itself
    def process_element(self, element):
        if self.element_checking_func(element):
            return {element: 1}
        return []

    def map_collection(self, text_collection):
        # some sanity checks
        if self.use_fixed_token_list and self.filter_type is not None:
            error("Specified token limiting but the token list is fixed.")

        # collection-wise information
        self.output_vectors, self.present_tokens, self.present_token_indexes = [], [], []

        # global token-wise frequencies
        with tictoc("Creating bow vectors"):
            for t, word_pos_list in enumerate(text_collection):
                debug("Text {}/{}".format(t + 1, len(text_collection)))
                text_token_freqs = {}
                present_words = []
                # for each word information chunk in the text
                for word_pos in word_pos_list:
                    # check and process the word
                    token_weights = self.element_processing_func(word_pos)
                    # if the word produced weights, add it to the present words
                    if not token_weights:
                        continue
                    present_words.append(word_pos[0])
                    # accumulate the results
                    for item, weight in token_weights.items():
                        text_token_freqs = self.update_token_frequency(item, weight, text_token_freqs)

                # completed document parsing
                # add required results
                self.output_vectors.append(text_token_freqs)
                word_list = [wp[0] for wp in word_pos_list]
                self.present_token_indexes.append([word_list.index(p) for p in present_words])
                self.present_tokens.append(len(present_words))
            # completed collection parsing - wrap up
            if not self.use_fixed_token_list:
                # if we computed the token list, store it
                self.token_list = list(self.global_freqs.keys())

        # apply filtering, if selected
        self.filter_tokens()
        info("Mapped the collection to frequencies of {} tokens.".format(len(self.token_list)))

    def get_weights(self):
        return self.output_vectors

    def get_global_weights(self):
        return self.global_freqs

    def get_present_token_indexes(self):
        return self.present_token_indexes

    def get_present_tokens(self):
        return self.present_tokens


class TFIDF(Bag):

    def __init__(self):
        Bag.__init__(self)
        self.calculate_global = True

    def map_collection(self, text_collection):
        Bag.map_collection(self, text_collection)
        # apply IDF
        self.idf_normalize()

    def idf_normalize(self, input_data=None):
        with tictoc("TFIDF normalization"):
            if input_data is not None:
                # just process the input
                self.output_vectors, self.global_freqs = input_data
            # normalize
            for vector_idx in range(len(self.output_vectors)):
                for token_key in self.output_vectors[vector_idx]:
                    self.output_vectors[vector_idx][token_key] = \
                        self.output_vectors[vector_idx][token_key] / self.global_freqs[token_key]

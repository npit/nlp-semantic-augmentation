import numpy as np
from utils import error, tictoc, debug


class Bag:
    token_list = None
    representation_dim = None

    # processes an element of an instance, producing pairs of vector_index and frequency_count
    def process_element(self, element):
        if element in self.token_list:
            return [(self.token2index[element], 1)]
        return []

    def map_collection(self, text_collection, token_list, calculate_global_frequencies=False):
        # collection-wise information
        self.output_vectors, self.present_tokens, self.present_token_indexes = [], [], []

        if token_list is not None:
            self.token_list = token_list
        elif self.token_list is None:
            error("No input neither stored token list for BoW.")

        self.num_tokens = len(self.token_list)
        self.token2index = {w: i for (w, i) in zip(self.token_list, list(range(len(self.token_list))))}
        self.representation_dim = len(token_list)
        # global token-wise frequencies
        self.global_freqs = np.zeros((self.num_tokens,), np.float32)

        with tictoc("Creating bow vectors"):
            for t, word_list in enumerate(text_collection):
                debug("Text {}/{}".format(t + 1, len(text_collection)))
                text_token_freqs = {}
                present_text_words = [w for w in word_list if w in self.token_list]
                for word in present_text_words:
                    coords_weights = self.process_element(word)
                    for word_index, weight in coords_weights:
                        if word not in text_token_freqs:
                            text_token_freqs[word_index] = weight
                        else:
                            text_token_freqs[word_index] += weight
                        # if it's the training set
                        if calculate_global_frequencies:
                            # accumulate DF
                            self.global_freqs[word_index] += weight

                self.output_vectors.append(text_token_freqs)
                self.present_token_indexes.append([word_list.index(p) for p in present_text_words])
                self.present_tokens.append(len(present_text_words))

    def get_vectors(self):
        return self.output_vectors
    def get_present_token_indexes(self):
        return self.present_token_indexes
    def get_present_tokens(self):
        return self.present_tokens


class TFIDF(Bag):
    def map_collection(self, text_collection, token_list):
        Bag.map_collection(self, text_collection, token_list, calculate_global_frequencies=True)
        # apply IDF
        self.idf_normalize()

    def idf_normalize(self):
        # prepare for element-wise division
        self.global_freqs[np.where(self.global_freqs == 0)] = 1
        # normalize
        for vector_idx in range(len(self.output_vectors)):
            for token_key in self.output_vectors[vector_idx]:
                self.output_vectors[vector_idx][token_key] = \
                    self.output_vectors[vector_idx][token_key] / self.global_freqs[token_key]

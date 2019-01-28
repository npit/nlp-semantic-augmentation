from utils import error, tictoc, debug, info
import defs
import tqdm


class Bag:
    term_list = None
    global_freqs = None
    output_vectors = None
    term_weighting_func = None
    term_greenlight_func = None
    use_fixed_term_list = None
    filter_type = None
    filter_quantity = None
    calculate_global = None
    term_delineation_func = None
    do_track_present_words = None

    def __init__(self):
        self.term_list = []
        self.global_freqs = {}
        self.use_fixed_term_list = False
        # non-fixed term list as the default
        self.term_weighting_func = self.unit_term_weighting
        self.term_greenlight_func = lambda x: True
        # words from a text, as default term delineation
        self.term_delineation_func = lambda x: [i for i in x]

        self.filter_type = None
        self.filter_quantity = None
        self.calculate_global = False

        self.do_track_present_words = True

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
            for doc_dict in self.output_vectors:
                for term, freq in doc_dict.items():
                    if freq >= self.filter_quantity:
                        terms_to_retain.add(term)
            self.delete_terms(terms_to_retain)

        elif self.filter_type == defs.limit.top:
            # sort w.r.t. collection-wise frequency, keep <threshold> first
            sorted_global_freqs = sorted(self.global_freqs, reverse=True, key=lambda tok: self.global_freqs[tok])
            terms_to_retain = sorted_global_freqs[:self.filter_quantity]
            self.delete_terms(terms_to_retain)
        else:
            error("Undefined limiting type: {}".format(self.filter_type))
        # limit the term list itself
        self.term_list = [tok for tok in self.term_list if tok in self.global_freqs]

    # remove terms from the accumulated collections
    def delete_terms(self, terms_to_retain):
        # do not delete terms if handling a fixed list
        for i, doc_dict in enumerate(self.output_vectors):
            self.output_vectors[i] = {tok: doc_dict[tok] for tok in doc_dict if tok in terms_to_retain}
        if self.calculate_global:
            self.global_freqs = {term: self.global_freqs[term] for term in terms_to_retain}

    # setters
    # #####################
    # enable term filtering
    def set_term_filtering(self, filter_type, filter_quantity):
        self.filter_type = filter_type
        self.filter_quantity = filter_quantity

    # set the term list of the bag
    def set_term_list(self, term_list):
        self.use_fixed_term_list = True
        self.term_list = list(term_list)
        self.global_freqs = {key: 0 for key in self.term_list}
        self.num_terms = len(self.term_list)
        self.term_greenlight_func = self.unit_term_weighting
        self.term_weighting_func = self.check_term_in_list

    # override term delineation function
    def set_term_delineation_function(self, func):
        self.term_delineation_func = func
        # present word tracking only available for the default (word-wise) delineation
        self.do_track_present_words = False

    # override default term processing function
    def set_term_weighting_function(self, func):
        self.term_weighting_func = func

    # override default term checking function
    def set_term_greenlight_function(self, func):
        self.term_greenlight_func = func

    # frequency updater function
    def update_term_frequency(self, item, weight, weights_dict):
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
        # some sanity checks
        if self.use_fixed_term_list and self.filter_type is not None:
            error("Specified term limiting but the term list is fixed.")

        # collection-wise information
        self.output_vectors, self.present_terms, self.present_term_indexes = [], [], []

        # global term-wise frequencies
        # with tqdm.tqdm(total=len(text_collection)"Creating bow vectors") as pbar:
        with tqdm.tqdm(desc="Creating bow vectors", total=len(text_collection), ascii=True) as pbar:
            for t, word_pos_list in enumerate(text_collection):
                pbar.set_description("Text {}/{}".format(t + 1, len(text_collection)))
                pbar.update()
                text_term_freqs = {}
                present_words = []
                # generate terms of interest
                # for each term-of-interest-chunk in the text
                for term_info in self.term_delineation_func(word_pos_list):
                    # process term and produce weight information of the processed result
                    term_weights = self.term_weighting_func(term_info)
                    # if the word produced weights, add it to the present words
                    if not term_weights:
                        continue
                    # add the term as a valid / present one
                    if self.do_track_present_words:
                        present_words.append(term_info)
                    # accumulate the term-weight results
                    for item, weight in term_weights.items():
                        text_term_freqs = self.update_term_frequency(item, weight, text_term_freqs)

                # completed document parsing
                # add required results
                self.output_vectors.append(text_term_freqs)
                if self.do_track_present_words:
                    word_list = [wp[0] for wp in word_pos_list]
                    self.present_term_indexes.append([word_list.index(p) for p in present_words])
                    self.present_terms.append(len(present_words))
            # completed collection parsing - wrap up
            if not self.use_fixed_term_list:
                # if we computed the term list, store it
                self.term_list = list(self.global_freqs.keys())

        # apply filtering, if selected
        self.filter_terms()
        info("Mapped the collection to frequencies of {} terms.".format(len(self.term_list)))

    # getters
    # #####################
    def get_weights(self):
        return self.output_vectors

    def get_global_weights(self):
        return self.global_freqs

    def get_present_term_indexes(self):
        return self.present_term_indexes

    def get_present_terms(self):
        return self.present_terms

    def get_term_list(self):
        return self.term_list


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
                for term_key in self.output_vectors[vector_idx]:
                    self.output_vectors[vector_idx][term_key] = \
                        self.output_vectors[vector_idx][term_key] / self.global_freqs[term_key]

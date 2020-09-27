from utils import error, tictoc, debug, info
import defs
import tqdm
import numpy as np
from scipy.sparse import csr_matrix

import defs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Bag:
    weighting = None
    tokenizer = None
    vocabulary = None
    threshold_type = None
    threshold = None

    def set_thresholds(self, threshold_type, threshold):
        """Populate thresholding"""
        self.threshold_type = threshold_type
        self.threshold = threshold

    def apply_thresholds(self, vectors):
        """Apply resholding"""
        sums = np.sum(vectors, axis=0)
        if self.threshold_type == defs.limit.frequency:
            term_idxs = np.where(sums > self.threshold)
        elif self.threshold_type == defs.limit.top:
            term_idxs = np.argsort(sums)[-self.threshold:]
        else:
            error(f"Undefined bag thresholding type: {self.threshold_type}")
        vectors = vectors.squeeze()[:, term_idxs]
        return vectors

    def __init__(self, weighting="counts", vocabulary=None, ngram_range=None):
        if weighting not in "bag tfidf".split():
            error(f"Undefined weighting {weighting}")
        self.weighting = weighting
        self.vocabulary = vocabulary
        if ngram_range is None:
            ngram_range = (1, 1)
        self.ngram_range = ngram_range

    def map_collection(self, text_collection, fit=True):
        mapper = CountVectorizer(tokenizer=self.tokenizer, vocabulary=self.vocabulary, ngram_range=self.ngram_range)
        if fit:
            # fit to discover the vocabulary
            vectors = mapper.fit_transform(text_collection)
        else:
            vectors = mapper.transform(text_collection)

        vectors = vectors.toarray()
        if self.threshold is not None:
            vectors = self.apply_thresholds(vectors)
        if len(vectors) == 0:
            vectors = np.zeros((0, len(self.vocabulary)), dtype=np.int32)
        else:
            if self.weighting == "tfidf" and len(vectors) > 0:
                tft = TfidfTransformer()
                vectors = tft.fit_transform(vectors).toarray()
        return vectors
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

    model = None

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

    def __init__(self, weighting="counts", vocabulary=None, ngram_range=None, tokenizer_func=None, analyzer=None):
        if weighting not in "bag tfidf".split():
            error(f"Undefined weighting {weighting}")
        self.weighting = weighting
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer_func
        self.analyzer = analyzer
        if ngram_range is None:
            ngram_range = (1, 1)
        self.ngram_range = ngram_range
        self.model = CountVectorizer(tokenizer=self.tokenizer, vocabulary=self.vocabulary, ngram_range=self.ngram_range, analyzer=self.analyzer)

    def get_vocabulary(self):
        return self.model.get_feature_names()

    def map_collection(self, text_collection, fit=False, transform=False):

        if fit:
            self.model.fit(text_collection)

        if transform:
            vectors = self.model.transform(text_collection).toarray()
            if self.threshold is not None:
                vectors = self.apply_thresholds(vectors)
            if len(vectors) == 0:
                vectors = np.zeros((0, len(self.vocabulary)), dtype=np.int32)
            else:
                if self.weighting == "tfidf" and len(vectors) > 0:
                    tft = TfidfTransformer()
                    vectors = tft.fit_transform(vectors).toarray()
            return vectors
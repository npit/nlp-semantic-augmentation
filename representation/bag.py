from utils import error, tictoc, debug, info
import defs
import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from functools import partial

from functools import partial
import defs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Bag:
    weighting = None
    tokenizer = None
    vocabulary = None
    min_counts = None

    model = None

    class PBarCountVectorizer(CountVectorizer):
        def __init__(self, **kwargs):
            CountVectorizer.__init__(self, **kwargs)

        def build_analyzer(self):
            analyzer = super().build_analyzer()
            updater_partial = partial(self.progbar_updater,analyzer_partial=analyzer)
            return updater_partial

        def progbar_updater(self, *args, **kwargs):
            self.pbar.update()
            analyzer_partial = kwargs["analyzer_partial"]
            del kwargs["analyzer_partial"]
            return analyzer_partial(*args, **kwargs)


    def set_min_features(self, threshold):
        """Populate thresholding"""
        self.min_counts = threshold

    def apply_thresholds(self, vectors):
        """Apply resholding"""
        if self.min_counts is not None:
            sums = np.sum(vectors, axis=0)
            term_idxs = np.where(sums > self.min_counts)
            vectors = vectors.squeeze()[:, term_idxs]
        else:
            # error(f"Undefined bag thresholding type: {self.threshold_type}")
            pass
        return vectors

    def __init__(self, weighting="counts", vocabulary=None, ngram_range=None, tokenizer_func=None, analyzer="word", max_terms=None):
        if weighting not in "bag tfidf".split():
            error(f"Undefined weighting {weighting}")
        self.weighting = weighting
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer_func
        self.analyzer = analyzer
        analyzer_arg = self.analyzer
        # analyzer_arg = self.analyzer_wrapper if callable(analyzer) else analyzer
        if ngram_range is None:
            ngram_range = (1, 1)
        self.ngram_range = ngram_range
        self.model = Bag.PBarCountVectorizer(tokenizer=self.tokenizer, vocabulary=self.vocabulary, ngram_range=self.ngram_range,
                                     analyzer=analyzer_arg, min_df=1, max_df=0.9, max_features=max_terms)

    def get_vocabulary(self):
        return self.model.get_feature_names()

    def map_collection(self, text_collection, fit=False, transform=False):

        if fit:
            with tqdm.tqdm(total=len(text_collection), desc="Fitting bag model", ascii=True) as pbar:
                self.model.pbar = pbar
                self.model.fit(text_collection)
        if transform:
            with tqdm.tqdm(total=len(text_collection), desc="Applying bag model", ascii=True) as pbar:
                self.model.pbar = pbar
                vectors = self.model.transform(text_collection).toarray()
            vectors = self.apply_thresholds(vectors)
            if len(vectors) == 0:
                vectors = np.zeros((0, len(self.vocabulary)), dtype=np.int32)
            else:
                if self.weighting == "tfidf" and len(vectors) > 0:
                    tft = TfidfTransformer()
                    vectors = tft.fit_transform(vectors).toarray()
            return vectors
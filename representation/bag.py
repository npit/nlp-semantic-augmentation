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
<<<<<<< Updated upstream
        pbar = None
        def __init__(self, **kwargs):
            CountVectorizer.__init__(self, **kwargs)

        # def _analyze(**kwargs):
        #     super()._analyze(**kwargs)

        def build_analyzer(self):
            """In-between function to update progress bar"""
            analyzer_partial = super().build_analyzer()
            return partial(self.progbar_updater, partial_obj=analyzer_partial, pbar=self.pbar)

        @staticmethod
        def progbar_updater(*args, **kwargs):
            kwargs["pbar"].update()
            partial_obj = kwargs["partial_obj"]
            del kwargs["pbar"]
            del kwargs["partial_obj"]
            return partial_obj(*args, **kwargs)
=======
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

>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
                                     analyzer=analyzer_arg, min_df=1, max_df=0.9, max_features=max_terms)
=======
                                     analyzer=self.analyzer, min_df=1, max_df=0.9, max_features=max_terms)


        
    # def default_analyzer(model):
    #     stop_words = model.get_stop_words()
    #     tokenize = model.build_tokenizer()
    #     model._check_stop_words_consistency(stop_words, preprocess,
    #                                         tokenize)
    #     return partial(model._analyze, ngrams=model._word_ngrams,
    #                     tokenizer=tokenize, preprocessor=model.preprocess,
    #                     decoder=model.decode, stop_words=stop_words)


    # def analyzer_wrapper(self, arg):
    #     self.pbar.update()
    #     self.analyzer(arg)
>>>>>>> Stashed changes

    def get_vocabulary(self):
        return self.model.get_feature_names()

    def map_collection(self, text_collection, fit=False, transform=False):

        if fit:
            with tqdm.tqdm(total=len(text_collection), desc="Fitting bag model", ascii=True) as pbar:
                self.model.pbar = pbar
                self.model.fit(text_collection)
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
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
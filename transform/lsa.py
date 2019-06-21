from sklearn.decomposition import TruncatedSVD

from transform.transform import Transform


class LSA(Transform):
    """Latent Semantic Analysis decomposition.

    Based on the truncated SVD implementation of sklearn.
    """
    base_name = "lsa"

    def __init__(self, representation):
        """LSA constructor"""
        Transform.__init__(self, representation)
        self.transformer = TruncatedSVD(self.dimension)
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform


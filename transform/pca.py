from sklearn.decomposition import PCA as skPCA
from transform.transform import Transform


class PCA(Transform):
    """Latent Semantic Analysis decomposition.

    Based on the truncated SVD implementation of sklearn.
    """
    base_name = "lsa"

    def __init__(self, config):
        """PCA constructor"""
        Transform.__init__(self, config)
        self.transformer = skPCA(self.dimension)
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform


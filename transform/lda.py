from sklearn.decomposition import LatentDirichletAllocation

from transform.transform import Transform


class LDA(Transform):
    """Latent Dirichlet Allocation transformation

    Uses the LDA implementation of sklearn.
    """
    base_name = "lda"

    def __init__(self, config):
        self.config = config
        Transform.__init__(self, config)
        self.transformer = LatentDirichletAllocation(n_components=self.dimension, random_state=self.config.misc.seed)
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform

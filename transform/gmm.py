
class GMMClustering(Transform):
    """Gausian Mixture Model clustering

    Uses the Gaussian mixture implementation of sklearn.
    """
    base_name = "gmm"

    def __init__(self, representation):
        """GMM-clustering constructor"""
        Transform.__init__(self, representation)
        self.transformer = GaussianMixture(self.dimension)
        self.process_func_train = self.fit_predict_proba
        self.process_func_test = self.transformer.predict_proba

    def fit_predict_proba(self, data):
        self.transformer = self.transformer.fit(data)
        return self.process_func_test(data)

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.means_

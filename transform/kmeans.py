from sklearn.cluster import KMeans

from transform.transform import Transform


class KMeansClustering(Transform):
    """KMeans clustering

    Uses the KMeans implementation of sklearn.
    """
    base_name = "kmeans"

    def __init__(self, config):
        """Kmeans-clustering constructor"""
        Transform.__init__(self, config)
        self.transformer = KMeans(self.dimension)
        self.process_func_train = self.fit
        self.process_func_test = self.do_transform

    def fit(self, data):
        self.transformer = self.transformer.fit(data)
        return self.do_transform(data)

    def do_transform(self, data):
        res = self.transformer.transform(data)
        return res

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.cluster_centers_


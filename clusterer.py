from learner import Learner
from utils import debug, error
import numpy as np
from sklearn.cluster import KMeans


class Clusterer(Learner):

    def __init__(self):
        """clusterer constructor"""
        self.sequence_length = 1
        Learner.__init__(self)

    def make(self, representation, dataset):
        max_epi = max([max(x) for x in representation.elements_per_instance])
        if max_epi != 1:
            error("Clustering requires single-vector instances, got {} max elements per instance.".format(max_epi))
        Learner.make(self, representation, dataset)


class KMeansClusterer(Clusterer):
    name = "kmeans"

    def __str__(self):
        return "name: {} clusters:{}".format(
            self.name, self.num_clusters)

    def __init__(self, config):
        """kmeans constructor"""
        self.config = config
        self.num_clusters = config.learner.num_clusters
        Clusterer.__init__(self)

    def make(self, representation, dataset):
        Clusterer.make(self, representation, dataset)

    # train a model on training & validation data portions
    def train_model(self, trainval_idx):
        # labels
        train_data, train_labels, _ = self.get_trainval_data(trainval_idx)
        # define the model
        model = KMeans(self.num_clusters)
        # train the damn thing!
        debug("Feeding the network train shapes: {} {}".format(train_data.shape, train_labels.shape))
        if val_datalabels is not None:
            debug("Using validation shapes: {} {}".format(*[v.shape if v is not None else "none" for v in val_datalabels]))
        model.fit(train_data)
        return model

    # evaluate a clustering
    def test_model(self, test_data, model):
        cluster_distances = model.transform(test_data)
        # convert to "similarity" scores
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

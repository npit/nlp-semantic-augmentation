import numpy as np
from sklearn.cluster import KMeans

from learning.learner import Learner
from utils import debug, error


class Clusterer(Learner):

    def __init__(self):
        """clusterer constructor"""
        self.sequence_length = 1
        self.num_clusters = self.config.num_clusters
        error("Attempted to create clusterer with a number of clusters equal to [{}]".format(self.num_clusters), self.num_clusters is None or self.num_clusters < 2)
        Learner.__init__(self)

    def make(self):
        Learner.make(self)
        try:
            if self.num_labels is not None:
                error("Specified number of clusters: {} not equal to the number of specified labels in the data: {}.".format(self.num_clusters, self.num_labels), self.num_clusters != self.num_labels)
        except:
            pass

    def is_supervised(self):
        """All clusterers don't require label information"""
        return False


class KMeansClusterer(Clusterer):
    name = "kmeans"

    def __str__(self):
        return "name: {} clusters:{}".format( self.name, self.num_clusters)

    def __init__(self, config):
        """kmeans constructor"""
        self.config = config
        Clusterer.__init__(self)

    def make(self):
        Clusterer.make(self)

    # train a model on training & validation data portions
    def train_model(self, train_index, embeddings, train_labels, val_index, val_labels):
        # define the model
        train_data = self.get_data_from_index(train_index, embeddings)
        model = KMeans(self.num_clusters)
        # train the damn thing!
        debug("Feeding the network train shapes: {}".format(train_data.shape))
        if val_index is not None:
            val_data = self.get_data_from_index(val_index, embeddings)
            debug("Using validation shapes: {}".format(val_data.shape))
        model.fit(train_data)
        return model

    # evaluate a clustering
    def test_model(self, test_index, embeddings, model):
        test_data = self.get_data_from_index(test_index, embeddings)
        cluster_distances = model.transform(test_data)
        # convert to "similarity" scores
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

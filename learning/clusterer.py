import numpy as np
from sklearn.cluster import KMeans
from bundle.datausages import *
from bundle.datausages import *

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
        """All clusterers don't require ground truth information"""
        return False
    def get_model(self):
        return self.model

    def set_component_outputs(self):
        """Set the output data of the clusterer"""
        super().set_component_outputs()
        # pass the clustering input features and the produced clusters for evaluation
        # self.validation.get_inputs_base_object(), self.validation.get_current_evaluation_indices()
        # pass the input data, and the indexes denoting on which of these the clusterer used to generate its predictions
        # (e.g. useful if validation occurred)
        dat = DataPack(Numeric(self.embeddings), Indices(self.prediction_indexes))
        self.data_pool.add_data_packs([dat], self.name)

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
    def train_model(self):
        # define the model
        train_data = self.get_data_from_index(self.train_index, self.embeddings)
        self.model = KMeans(self.num_clusters)
        # train the damn thing!
        debug("Feeding the network train shapes: {}".format(train_data.shape))
        if self.val_index is not None and self.val_index.size > 0:
            val_data = self.get_data_from_index(self.val_index, self.embeddings)
            debug("Using validation shapes: {}".format(val_data.shape))
        self.model.fit(train_data)
        return self.model

    # evaluate a clustering
    def test_model(self, model):
        test_data = self.get_data_from_index(self.test_index, self.embeddings)
        cluster_distances = model.transform(test_data)
        # convert to "similarity" scores
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

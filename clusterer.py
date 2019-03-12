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
        train_labels, val_labels = self.prepare_labels(trainval_idx)
        # data
        if self.num_train != self.num_train_labels:
            trainval_idx = self.expand_index_to_sequence(trainval_idx)
        train_data, val_data = [self.process_input(data) if len(data) > 0 else np.empty((0,)) for data in
                                [self.train[idx] if len(idx) > 0 else [] for idx in trainval_idx]]
        val_datalabels = (val_data, val_labels) if val_data.size > 0 else None
        # build model
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

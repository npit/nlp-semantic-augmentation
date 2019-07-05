from learning.dnn import MLP, LSTM
from learning.clusterer import KMeansClusterer
from learning.classifier import NaiveBayes
from utils import error


class Instantiator:
    name = "learner"

    def create(config):
        """Function to instantiate a learning"""
        name = config.learner.name
        candidates = [MLP, LSTM, KMeansClusterer, NaiveBayes]
        for candidate in candidates:
            if name == candidate.name:
                return candidate(config)
        error("Undefined learning: {}".format(name))

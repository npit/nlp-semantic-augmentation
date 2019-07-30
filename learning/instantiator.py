from learning.classifier import Dummy, NaiveBayes
from learning.clusterer import KMeansClusterer
from learning.dnn import LSTM, MLP, BiLSTM
from utils import error


class Instantiator:
    component_name = "learner"

    def create(config):
        """Function to instantiate a learning"""
        name = config.learner.name
        candidates = [MLP, LSTM, BiLSTM, KMeansClusterer, NaiveBayes, Dummy]
        for candidate in candidates:
            if name == candidate.name:
                return candidate(config)
        error("Undefined learning: {}".format(name))

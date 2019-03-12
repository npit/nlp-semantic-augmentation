from dnn import MLP, LSTM
from clusterer import KMeansClusterer
from utils import error


def instantiate_learner(config):
    """Function to instantiate a learner"""
    name = config.learner.name
    candidates = [MLP, LSTM, KMeansClusterer]
    for candidate in candidates:
        if name == candidate.name:
            return candidate(config)
    error("Undefined learner: {}".format(name))

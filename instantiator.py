from dnn import MLP, LSTM
from utils import error


def instantiate_learner(config):
    """Function to instantiate a learner"""
    name = config.learner.name
    candidates = [MLP, LSTM]
    for candidate in candidates:
        if name == candidate.name:
            return candidate(config)
    error("Undefined learner: {}".format(name))

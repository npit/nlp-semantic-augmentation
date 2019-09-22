from learning.classifier import Dummy, NaiveBayes, LogisticRegression
from learning.clusterer import KMeansClusterer
from learning.dnn import LSTM, MLP, BiLSTM
from learning.neural.embedding import EmbeddingNN
from utils import error


class Instantiator:
    component_name = "learner"

    @staticmethod
    def create(config):
        """Function to instantiate a learning"""
        name = config.learner.name
        candidates = [MLP, LSTM, BiLSTM, KMeansClusterer, NaiveBayes, Dummy, EmbeddingNN, LogisticRegression]
        for candidate in candidates:
            if name == candidate.name:
                return candidate(config)
        error("Undefined learning: {}. Available ones are: {}".format(name, candidates))

from learning.classifier import Dummy, LogisticRegression, NaiveBayes
from learning.clusterer import KMeansClusterer
# from learning.dnn import MLP as tfMLP
from utils import error, info

from learning.neural.models import instantiator as neural_instantiator

class Instantiator:
    component_name = "learner"

    @staticmethod
    def create(config):
        """Function to instantiate a learning"""
        name = config.name
        candidates = [KMeansClusterer, NaiveBayes, Dummy, LogisticRegression]
        # instantiate non-neural candidates
        for candidate in candidates:
            if name == candidate.name:
                return candidate(config)

        # instantiate neural candidates
        try:
            neural_wrapper_class = neural_instantiator.get_neural_wrapper_class(name)
            info(f"Parsed wrapper: {neural_wrapper_class.name} from learner name: {name}")
            return neural_wrapper_class(config)
        except ValueError:
            # handled in the neural instantiator
            pass
        error("Undefined learning: {}. Available ones are: {}".format(name, candidates))

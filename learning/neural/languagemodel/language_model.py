"""Module for the incorporation of pretrained language models"""
from learning.supervised_learner import SupervisedLearner
from utils import error


class LanguageModel(SupervisedLearner):
    """Class to implement a neural language model"""
    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def acquire_embedding_information(self):
        # override input acquisition from the input bundle
        self.make_language_model()
        # require text
        error(f"{self.name} requires input texts", not self.inputs.has_text())
        self.map_text()

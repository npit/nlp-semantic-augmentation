"""Module for the incorporation of pretrained language models"""
from learning.neural.languagemodel import NeuralLanguageModel
import defs
from utils import error, info, one_hot
import numpy as np


class LabelledNeuralLanguageModel(NeuralLanguageModel):
    """Class to implement a neural language model that ingests text sequences and labels"""

    def __init__(self):
        """
        Constructor
        """
        super().__init__(self.config)

    def build_model(self):
        """In language models, the neural model is already built to generate embeddings from text"""
        pass

    def acquire_embedding_information(self):
        """Embedding acquisition for language models"""
        self.configure_language_model(self.inputs.get_labels().labelset)
        super().acquire_embedding_information()
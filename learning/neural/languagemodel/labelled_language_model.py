"""Module for the incorporation of pretrained language models"""
from learning.neural.dnn import LabelledDNN
from learning.neural.languagemodel.language_model import NLM
import defs
from utils import error, info, one_hot
import numpy as np


class LabelledNLM(NLM, LabelledDNN):
    """Class to implement a neural language model that ingests text sequences and labels"""
    name = "labelled_nlm"

    def __init__(self):
        """
        Constructor
        """
        LabelledDNN.__init__(self.config)

    def build_model(self):
        self.neural_model = self.neural_model_class(self.config, self.num_labels)

    def acquire_embedding_information(self):
        """Embedding acquisition for language models"""
        self.configure_language_model(self.inputs.get_labels().labelset)
        super().acquire_embedding_information()

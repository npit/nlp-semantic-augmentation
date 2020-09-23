"""Module for the incorporation of transformer models from huggingface"""
import numpy as np

import defs
import torch
from utils import error, info, one_hot
from learning.neural.languagemodel.language_model import NeuralLanguageModel

from torch.utils.data import DataLoader
from learning.neural.models import instantiator
from os.path import exists, dirname

class HuggingfaceTransformerLanguageModel(NeuralLanguageModel):
    """Wrapper class for huggingface transformer models"""

    name = "huggingface_labelled_transformer_lm"
    use_pretrained = True
    model = None

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.config = config
        NeuralLanguageModel.__init__(self)

    def configure_language_model(self, labelset):
        self.num_labels = len(labelset)
        super().configure_language_model()

    def get_model(self):
        return self.neural_model_class(self.config, self.num_labels, use_pretrained=True)
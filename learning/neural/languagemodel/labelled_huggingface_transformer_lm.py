"""Module for the incorporation of transformer models from huggingface"""
import numpy as np

import defs
import torch
from utils import error, info, one_hot
# from learning.neural.languagemodel.labelled_language_model import LabelledNLM
from learning.neural.dnn import LabelledDNN
from learning.neural.languagemodel.huggingface_transformer_lm import HuggingfaceTransformer

from torch.utils.data import DataLoader
from learning.neural.models import instantiator
from os.path import exists, dirname

class LabelledHuggingfaceTransformer(HuggingfaceTransformer, LabelledDNN):
    """Wrapper class for huggingface classifiers"""

    name = "hf_labelled_transformer"
    use_pretrained = True
    model = None

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.config = config
        LabelledDNN.__init__(self, config)
        HuggingfaceTransformer.__init__(self, config)

    def build_model(self):
        self.neural_model = self.neural_model_class(self.config, self.num_labels)

    def get_model(self):
        return self.neural_model_class(self.config, self.num_labels, use_pretrained=True)

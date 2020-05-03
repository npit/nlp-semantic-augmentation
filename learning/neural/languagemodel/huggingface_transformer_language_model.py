"""Module for the incorporation of transformer models from huggingface"""
import numpy as np

import defs
import torch
from utils import error, info, one_hot
from learning.neural.languagemodel.language_model import SupervisedNeuralLanguageModel

from torch.utils.data import DataLoader
from learning.neural.models import instantiator
from os.path import exists, dirname

class HuggingfaceTransformerLanguageModel(SupervisedNeuralLanguageModel):
    """Wrapper class for huggingface transformer models"""

    name = "huggingface_language_model"
    use_pretrained = True
    model = None

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.config = config
        SupervisedNeuralLanguageModel.__init__(self)

    def configure_language_model(self, labelset):
        self.num_labels = len(labelset)
        self.neural_model = self.get_model()
        self.tokenizer = self.get_tokenizer()

    def get_model(self):
        return self.neural_model_class(self.config, self.num_labels, use_pretrained=True)

    def get_tokenizer(self):
        return self.neural_model.get_tokenizer()


    # handle huggingface models IO
    def save_model(self, model):
        path = dirname(self.get_current_model_path())
        info("Saving model to {}".format(path))
        # access the huggingface class itself
        self.neural_model.model.save_pretrained(path)

    def load_model(self):
        path = self.get_current_model_path()
        if not path or not exists(path):
            return None
        info("Loading existing learning model from {}".format(path))
        model = self.neural_model_class.huggingface_model_class.from_pretrained(path)
        return model


    def encode_text(self, text):
        """Encode text into a sequence of tokens"""
        # apply the sequence length limit
        text = " ".join(text.split()[:self.sequence_length])
        encoded = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).numpy()
        return encoded

    def make(self):
        super().make()


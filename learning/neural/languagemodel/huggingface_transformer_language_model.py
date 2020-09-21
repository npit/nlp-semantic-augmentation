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

    def map_text(self):
        """Process input text into tokenized elements"""
        super().map_text()
        # associate the attention masks with the model, after their computation
        self.neural_model.configure_masking(self.masks)

    # handle huggingface models IO
    def save_model(self, model):
        path = dirname(self.get_model_path())
        info("Saving model to {}".format(path))
        # access the huggingface class itself
        self.neural_model.model.save_pretrained(path)

    def load_model(self):
        path = self.get_model_path()
        if not path or not exists(path):
            return None
        info("Loading existing learning model from {}".format(path))
        model = self.neural_model_class.huggingface_model_class.from_pretrained(path)
        return model


    def encode_text(self, text):
        """Encode text into a sequence of tokens
        Input has to be a multi-token text
        """
        # Tokenize sentence and add `[CLS]` and `[SEP]` tokens.
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True,
            max_length=self.sequence_length, return_attention_mask=True, return_tensors='pt',
                                                pad_to_max_length=True, truncation=True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return input_ids, attention_mask

        # # apply the sequence length limit
        # text = " ".join(text.split()[:self.sequence_length])
        # encoded = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).numpy()
        # return encoded

    def make(self):
        super().make()


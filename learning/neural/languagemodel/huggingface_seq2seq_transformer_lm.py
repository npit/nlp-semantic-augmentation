"""Module for the incorporation of transformer models from huggingface"""
import numpy as np

import defs
import logging
import torch
from utils import error, info, one_hot, equal_lengths, warning, shapes_list
from learning.neural.languagemodel.huggingface_transformer_lm import HuggingfaceTransformerLanguageModel
from bundle.datatypes import *
from bundle.datausages import *

from torch.utils.data import DataLoader
from learning.neural.models import instantiator
from os.path import exists, dirname, join

from transformers import EncoderDecoderConfig, EncoderDecoderModel

class HuggingfaceSeq2SeqTransformerLanguageModel(HuggingfaceTransformerLanguageModel):
    """Wrapper class for seq2seqhuggingface transformer models"""

    name = "huggingface_seq2seq_transformer_lm"

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.config = config
        self.sequence_length = self.config.sequence_length
        HuggingfaceTransformerLanguageModel.__init__(self, config)


    def get_model(self):
        return self.neural_model_class(self.config, sequence_length=self.config.sequence_length, use_pretrained=True)

    # def fetch_language_model_inputs(self):
    #     # obtain regular texts
    #     super().fetch_language_model_inputs()
    #     # obtain target texts as well
          # MOVED TO SUP. LEARNER
    #     # number of index groups have to match
    #     error("Unequal indices for input and target texts", not equal_lengths(self.indices.instances, self.target_indices.instances))
    def get_ground_truth(self):
        """Ground truth retrieval function"""
        # fetch the gt textual gt token embeddings
        return self.target_embeddings, self.target_masks

    def load_model(self):
        try:
            # get neural model 
            info(f"Attempting load of prebuilt s2s LM: {self.get_full_name()}")
            model = self.neural_model_class.get_from_pretrained(self.get_model_path())
            self.neural_model = self.neural_model_class(self.config, sequence_length=self.config.sequence_length, pretrained_model=model)
            self.model = self.neural_model
        except Exception as ex:
            return False
        return True 

    def get_train_test_targets(self):
        test = self.target_embeddings[self.target_test_embedding_index] if self.target_test_embedding_index else None
        return (self.target_embeddings[self.target_train_embedding_index], test)

    def map_text(self):
        """Process input text into tokenized elements"""
        # map regular inputs
        super().map_text()
        # map targets
        info(f"Tokenizing seq2seq LM textual ground truth data to tokens with a sequence length of {self.sequence_length}")

        self.target_embeddings, self.target_masks, self.target_train_embedding_index, self.target_test_embedding_index \
            = self.map_text_collection(self.targets, self.target_indices)

        # check correspondence with inputs
        checks = [((self.target_embeddings, self.embeddings, self.masks, self.target_masks), "embeddings and masks"),
                  ((self.train_embedding_index, self.target_train_embedding_index), "train indexes"),
                  ((self.test_embedding_index, self.target_test_embedding_index), "test indexes")]
        error_exists = False
        for ch in checks:
            if not equal_lengths(ch[0]):
                warning(ch[1] + "shapes:" + shapes_list(ch[0]))
                error_exists = True
            error("Inconsistent inputs / targets mapping outputs:", error_exists)

    def set_component_outputs(self):
        super().set_component_outputs()
        info("Converting output sequence tokens to string")
        # convert predictions to text as well
        self.text_predictions = []
        for prediction_set in self.predictions:
            text_preds = []
            for predictions_instance in prediction_set:
                txt = self.prediction_to_text(predictions_instance)
                text_preds.append(txt)
            self.text_predictions.append(text_preds)

        # (e.g. useful if validation occurred)
        dat = DataPack(Text(self.text_predictions), Predictions([np.arange(len(self.predictions))]))
        self.data_pool.add_data_packs([dat], self.name)

    def prediction_to_text(self, pred):
        """Convert a sequence of predicted token ids to the corresponding text"""

        toks = self.tokenizer.convert_ids_to_tokens(pred, skip_special_tokens=True)
        return " ".join(toks)




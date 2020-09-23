from learning.neural.base_model import BaseModel
from transformers import (BertConfig, BertForSequenceClassification, BertModel, BertTokenizer)
from learning.neural.models.huggingface_classifier import HuggingfaceSequenceClassifier
import logging


class Bert(HuggingfaceSequenceClassifier):
    """Class for the BERT language model"""
    name = "bert"
    huggingface_model_class = BertForSequenceClassification

    def __init__(self, config, num_labels, use_pretrained=True):
        """Constructor"""
        self.num_labels = num_labels
        self.use_pretrained = use_pretrained
        self.pretrained_id = "bert-base-uncased"
        super().__init__(config)
        config_args = {"pretrained_model_name_or_path": self.pretrained_id, "num_labels": self.num_labels}
        # suspend logging
        lvl = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARN)
        if use_pretrained:
            model = BertForSequenceClassification.from_pretrained(self.pretrained_id, num_labels=num_labels,
                                                                  output_hidden_states=False, output_attentions=False)
        else:
            model = BertForSequenceClassification(BertConfig(num_labels=num_labels,
                                                    output_hidden_states=False,
                                                    output_attentions=False))
        logging.getLogger().setLevel(lvl)
        self.model = model

    def get_tokenizer(self):
        if self.use_pretrained:
            tokenizer = BertTokenizer.from_pretrained(self.pretrained_id)
        else:
            tokenizer = BertTokenizer(BertConfig(num_labels=self.num_labels))
        return tokenizer
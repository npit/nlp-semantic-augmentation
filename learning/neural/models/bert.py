from transformers import (BertConfig, BertForSequenceClassification, BertModel, BertTokenizer)
from learning.neural.models.huggingface_classifier import HuggingfaceSequenceClassifier
import logging


class Bert(HuggingfaceSequenceClassifier):
    """Class for the BERT language model"""
    name = "bert"
    huggingface_model_class = BertForSequenceClassification
    pretrained_id = "bert-base-uncased"

    def __init__(self, config, num_labels, use_pretrained=True):
        """Constructor"""
        super().__init__(config, num_labels)
        self.use_pretrained = use_pretrained
        if config.model_id is not None:
            self.pretrained_id = config.model_id
        else:
            self.pretrained_id = "bert-base-uncased"
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

    @classmethod
    def get_tokenizer(config, use_pretrained=True):
        if use_pretrained:
            tokenizer = BertTokenizer.from_pretrained(Bert.pretrained_id)
        else:
            tokenizer = BertTokenizer(BertConfig(num_labels=self.num_labels))
        return tokenizer

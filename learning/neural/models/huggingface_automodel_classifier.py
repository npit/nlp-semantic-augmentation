from transformers import AutoConfig, AutoModel, AutoTokenizer
from learning.neural.models.huggingface_classifier import HuggingfaceSequenceClassifier
import logging
from utils import error


class HF_Automodel(HuggingfaceSequenceClassifier):
    """Class for the BERT language model"""
    name = "huggingface_automodel"
    pretrained_id = None

    def __init__(self, config, num_labels):
        """Constructor"""
        super().__init__(config, num_labels)
        error("Need a HF model id", config.model_id is None)
        self.pretrained_id = config.model_id

        # suspend logging
        lvl = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARN)
        self.model_config = AutoConfig.from_pretrained(self.pretrained_id, num_labels=num_labels, output_hidden_states=False, output_attentions=False)
        model = AutoModel.from_pretrained(self.pretrained_id, config=self.model_config)
        self.model = model

        logging.getLogger().setLevel(lvl)

    @classmethod
    def get_tokenizer(cls, config):
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        return tokenizer

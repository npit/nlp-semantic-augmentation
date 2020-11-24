from learning.neural.base_model import BaseModel
from transformers import EncoderDecoderModel, EncoderDecoderConfig
from transformers import (BertConfig, BertForNextSentencePrediction, BertModel, BertTokenizer)
from learning.neural.models.huggingface_seq2seq import HuggingfaceSeq2seq
import logging
from os.path import exists, dirname, join
from utils import info


class BertSeq2Seq(HuggingfaceSeq2seq):
    """Class for the BERT language model"""
    name = "bert_seq2seq"
    huggingface_model_class = EncoderDecoderModel

    # pretrained_id = "bert-base-uncased"
    pretrained_id = "nlpaueb/bert-base-greek-uncased-v1"

    @staticmethod
    def get_from_pretrained(path):
        conf_path = join(dirname(path), "config.json")
        conf = EncoderDecoderConfig.from_pretrained(conf_path)
        model = EncoderDecoderModel.from_pretrained(path, config=conf)
        return model

    def __init__(self, config, sequence_length, use_pretrained=True, pretrained_model=None):
        """Constructor"""
        super().__init__(config, sequence_length)
        # suspend logging due to hellish verbosity
        lvl = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARN)
        config_args = {"pretrained_model_name_or_path": self.pretrained_id}

        if pretrained_model is None:
            if use_pretrained:
                model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.pretrained_id, self.pretrained_id)
            else:
                enc, dec = BertConfig(), BertConfig()
                dec.is_decoder = True
                dec.add_cross_attention = True
                enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
                model = EncoderDecoderModel(config=enc_dec_config)

            logging.getLogger().setLevel(lvl)
            self.model = model
        else:
            self.model = pretrained_model
        logging.getLogger().setLevel(self.config.print.log_level.upper())

    def get_tokenizer(use_pretrained=True, pretrained_id=None):
        info("Fetching tokenizer: " + ("id=" + str(pretrained_id)) if use_pretrained else " UNTRAINED!")
        if pretrained_id is None:
            pretrained_id = BertSeq2Seq.pretrained_id
        if use_pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained_id)
        else:
            tokenizer = BertTokenizer(BertConfig())
        return tokenizer

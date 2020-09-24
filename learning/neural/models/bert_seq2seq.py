from learning.neural.base_model import BaseModel
from transformers import EncoderDecoderModel, EncoderDecoderConfig
from transformers import (BertConfig, BertForNextSentencePrediction, BertModel, BertTokenizer)
from learning.neural.models.huggingface_seq2seq import HuggingfaceSeq2seq
import logging


class BertSeq2Seq(HuggingfaceSeq2seq):
    """Class for the BERT language model"""
    name = "bert_seq2seq"
    huggingface_model_class = BertForNextSentencePrediction

    def __init__(self, config, sequence_length, use_pretrained=True):
        """Constructor"""
        self.use_pretrained = use_pretrained
        self.pretrained_id = "bert-base-uncased"
        # greek
        self.pretrained_id = "nlpaueb/bert-base-greek-uncased-v1"
        super().__init__(config, sequence_length)
        # suspend logging
        lvl = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARN)



        config_args = {"pretrained_model_name_or_path": self.pretrained_id}

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

    def get_tokenizer(self):
        if self.use_pretrained:
            tokenizer = BertTokenizer.from_pretrained(self.pretrained_id)
        else:
            tokenizer = BertTokenizer(BertConfig())
        return tokenizer

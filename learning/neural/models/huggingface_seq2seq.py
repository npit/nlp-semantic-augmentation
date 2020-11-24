import torch
from torch.nn import functional as F
from learning.neural.base_model import BaseModel
from utils import error, info, debug

from transformers import EncoderDecoderConfig, EncoderDecoderModel
class HuggingfaceSeq2seq(BaseModel):
    """Neural model provided by huggingface"""

    use_pretrained = True
    # specify the wrapper class name for huggingface models
    wrapper_name = "hf_seq2seq_transformer"

    masks = None

    def __init__(self, config, sequence_length):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.sequence_length = sequence_length
        super().__init__(config, self.wrapper_name, config.folders.run, self.name)

    def make_predictions(self, inputs):
        # generate
        debug(f"Making predictions on input data {inputs.shape} and seqlen {self.sequence_length}")
        self.model.to(self.device_name)
        input_tokens = torch.LongTensor(self.embeddings[inputs]).to(self.device_name)
        att_mask = torch.LongTensor(self.masks[inputs]).to(self.device_name)
        preds =  self.model.generate(input_tokens, decoder_start_token_id=self.model.config.decoder.pad_token_id,
            max_length=self.sequence_length, min_length=self.sequence_length, attention_mask=att_mask)
        return preds

    def assign_ground_truth(self, gt):
        """Ground truth initialization"""
        self.ground_truth, self.ground_truth_mask = gt

    def compute_loss(self, logits, y):
        """Return a loss estimate for the predictions"""
        # return F.nll_loss(logits, y)
        return self.current_loss

    def configure_masking(self, masks):
        """Assign mask information to the model"""
        self.masks = masks

    def forward(self, inputs):
        """Huggingface model forward pass"""
        if len(inputs) < self.config.train.batch_size:
            x = torch.zeros(self.config.train.batch_size, dtype=torch.long, requires_grad=False)
            x[:len(inputs)] = inputs
            inputs = x
            # print("Padded:", inputs)
        input_tokens = torch.LongTensor(self.embeddings[inputs, :]).to(self.device_name)
        input_mask = torch.LongTensor(self.masks[inputs, :]).to(self.device_name)
        input_labels = torch.LongTensor(self.ground_truth[inputs, :]).to(self.device_name)
        outputs = self.model(input_ids=input_tokens, decoder_input_ids=input_tokens, labels=input_labels, attention_mask=input_mask)
        self.current_loss = outputs[0]
        logits = outputs[1]
        return logits

    def get_data(self, index):
        """Fetch embedding index """
        return torch.LongTensor(self.embeddings[index])

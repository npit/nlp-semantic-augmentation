import torch
from torch.nn import functional as F
from learning.neural.base_model import BaseModel
from utils import error, info

class HuggingfaceSeq2seq(BaseModel):
    """Neural model provided by huggingface"""

    use_pretrained = True
    ground_truth_embeddings = None
    # specify the wrapper class name for huggingface models
    wrapper_name = "huggingface_seq2seq_transformer_lm"

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        super().__init__(config, self.wrapper_name, config.folders.run, self.name)

    def make_predictions(self, inputs):
        # generate
        inputs = torch.LongTensor(self.embeddings[inputs])
        preds =  self.model.generate(inputs, decoder_start_token_id=self.model.config.decoder.pad_token_id)
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
            print("Padded:", inputs)
        input_tokens = torch.LongTensor(self.embeddings[inputs, :])
        input_mask = torch.LongTensor(self.masks[inputs, :])
        input_labels = torch.LongTensor(self.ground_truth[inputs, :])
        output = self.model(input_ids=input_tokens, decoder_input_ids=input_tokens, labels=input_labels, return_dict=True)
        self.current_loss = output.loss
        return output.logits

    def get_data(self, index):
        """Fetch embedding index """
        return torch.LongTensor(self.embeddings[index])

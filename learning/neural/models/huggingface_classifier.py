import torch
from learning.neural.base_model import BaseModel
from utils import error, info

class HuggingfaceSequenceClassifier(BaseModel):
    """Neural model provided by huggingface"""

    use_pretrained = True
    # specify the wrapper class name for huggingface models

    wrapper_name = "hf_labelled_transformer"

    masks = None

    def __init__(self, config, num_labels):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.num_labels = num_labels
        super().__init__(config, self.wrapper_name, config.folders.run, self.name)


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
        # print("input idx shp:", inputs.shape)
        # print("embeddings shp:", self.embeddings.shape)
        input_tokens = torch.LongTensor(self.embeddings[inputs, :])
        input_mask = torch.LongTensor(self.masks[inputs, :])
        # print("input tokens shp:", input_tokens.shape)
        logits = self.model(input_tokens, attention_mask=input_mask)[0]
        # print(logits)
        # print("Check https://mc.ai/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic/ for potential training required stuff")
        return logits

    def get_data(self, index):
        """Fetch embedding index """
        return torch.LongTensor(self.embeddings[index])

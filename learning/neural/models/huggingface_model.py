import torch
from learning.neural.base_model import BaseModel
from utils import error, info

class HuggingfaceModel(BaseModel):
    """Neural model provided by huggingface"""

    use_pretrained = True
    # specify the wrapper class name for huggingface models
    wrapper_name = "huggingface_language_model"

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        super().__init__(config, self.wrapper_name)

    def forward(self, inputs):
        """Huggingface model forward pass"""
        if len(inputs) < self.config.train.batch_size:
            x = torch.zeros(self.config.train.batch_size, dtype=torch.long, requires_grad=False)
            x[:len(inputs)] = inputs
            inputs = x
            print("Padded:", inputs)
        print("input idx shp:", inputs.shape)
        print("embeddings shp:", self.embeddings.shape)
        input_tokens = torch.LongTensor(self.embeddings[inputs, :])
        print("input tokens shp:", input_tokens.shape)
        logits = self.model(input_tokens)[0]
        # print(logits)
        # print("Check https://mc.ai/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic/ for potential training required stuff")
        return logits

    def get_data(self, index):
        """Fetch embedding index """
        return torch.LongTensor(self.embeddings[index])

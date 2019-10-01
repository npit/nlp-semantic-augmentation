import torch

from learning.neural.model import BaseModel
from learning.neural.neuralnet import NeuralNet
from torch.nn import functional as F


class MLPModel(BaseModel):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, train_embedding, num_labels):
        BaseModel.__init__(self)
        """Model builder function for dense MLP"""
        self.build_embedding_layer(num_embeddings, embedding_dim, train_embedding)
        current_dim = embedding_dim
        # build chain of dense layers
        layers = []
        for dim  in hidden_dim:
            layers.append(torch.nn.Linear(current_dim, dim))
            current_dim = dim
        
        self.linear_layers = torch.nn.ModuleList(layers)
        # build final classification layer
        self.linear_out = torch.nn.Linear(current_dim, num_labels)

    def forward(self, input_data):
        """Forward pass method"""
        # embedding output
        data = self.embedding_layer(input_data)
        # dense chain
        for layer in self.linear_layers:
            data = F.dropout(F.relu(layer(data)),p=0.3)
        # classification
        return self.linear_out(data)


class MLP(NeuralNet):
    name = "tmlp"
    """Generic MLP"""
    def __init__(self, config):
        NeuralNet.__init__(self, config)

    def get_model(self):
        return MLPModel(len(self.embeddings), self.embeddings.shape[-1], self.hidden_dim, self.train_embedding, self.num_labels)

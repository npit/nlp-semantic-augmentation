import torch

from learning.neural.model import BaseModel
from learning.neural.neuralnet import NeuralNet


class MLPModel(BaseModel):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, train_embedding, num_labels):
        BaseModel.__init__(self)
        """Model builder function for dense MLP"""
        self.build_embedding_layer(num_embeddings, embedding_dim, train_embedding)
        current_dim = embedding_dim
        self.layers = []
        for idx, _  in enumerate(hidden_dim):
            self.layers.append(torch.nn.Linear(current_dim, hidden_dim[idx]))
            current_dim = hidden_dim[idx]
        self.build_softmax(current_dim, num_labels)

    def forward(self, input_data):
        """Forward pass method"""
        # embedding output
        data = self.embedding_layer(input_data)
        # dense chain
        for layer in self.layers:
            data = self.sigmoid(layer(data))
        # classification
        return self.softmax(self.sigmoid(self.linear_out(data)))


class MLP(NeuralNet):
    name = "tMLP"
    """Generic MLP"""
    def __init__(self, config):
        NeuralNet.__init__(self, config)

    def get_model(self):
        return MLPModel(len(self.embeddings), self.embeddings.shape[-1], self.hidden_dim, self.train_embedding, self.num_labels)

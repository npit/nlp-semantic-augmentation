import torch

from learning.neural.neuralnet import NeuralNet


class BaseModel(torch.nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def build_embedding_layer(self, num_embeddings, embedding_dim, train_embedding):
        # build the model
        self.embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        if not train_embedding:
            self.embedding_layer.requires_grad = False

    def build_softmax(self, input_dim, num_labels):
        # build the model
        self.linear_out = torch.nn.Linear(input_dim, num_labels)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

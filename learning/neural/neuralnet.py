import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from learning.classifier import Classifier
from utils import debug, error, one_hot

"""Base torch-based neural model with learnable input embeddings"""
class NeuralNet(Classifier):

    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.config = config
        Classifier.__init__(self)

    def make(self):
        Classifier.make(self)
        self.num_layers = self.config.learner.num_layers
        self.hidden_dim = self.config.learner.hidden_dim
        if type(self.hidden_dim) is int:
            self.hidden_dim = [self.hidden_dim] * self.num_layers
        self.model_class = self.config.learner.name

    def get_model(self):
        error("Attempted to access base neuralnet model getter.")
        return None

    def train_model(self, train_index, embeddings, train_labels, val_index, val_labels):
        """Training method for pytorch networks"""
        model = self.get_model()
        # train_data = self.get_data_from_index(train_index, embeddings)
        if val_index is not None:
            val_data = self.get_data_from_index(train_index, embeddings)

        # setup vars
        optimizer = optim.Adam(model.parameters())
        # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()

        # training loop
        train_set = NeuralNet.TorchDataset(train_index, train_labels)
        data_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch_index, index_labels in enumerate(data_loader):

                index, labels = index_labels
                optimizer.zero_grad()

                preds = model.forward(index)
                loss = loss_func(preds, labels)
                loss.backward()
                optimizer.step()
                debug("Epoch {}, batch {}, loss {:.3f}".format(epoch+1, batch_index+1, loss.item()))
        return model

    def test_model(self, test_index, embeddings, model):
        test_set = NeuralNet.TorchDataset(test_index)
        data_loader = DataLoader(test_set, batch_size=self.batch_size)
        predictions = np.ndarray((0, self.num_labels), np.float32)
        with torch.no_grad():
            model.eval()
            # disable learning
            for data_index in data_loader:
                preds = model.forward(data_index)
                predictions = np.append(predictions, preds, axis=0)
        return predictions

    class TorchDataset(Dataset):
        data = None
        labels = None
        """Dataset class for easy batching with pytorch"""
        def __init__(self, data, labels=None):
            self.data = data
            if labels is not None:
                self.labels = labels
        def __getitem__(self, idx):
            if self.labels is not None:
                return (self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long))
            return self.data[idx]
        def __len__(self):
            return len(self.data)
        

# torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)

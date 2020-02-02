from copy import deepcopy

import numpy as np

import torch
from learning.classifier import Classifier
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utils import debug, error


class NeuralNet(Classifier):
    """Base torch-based neural model with learnable input embeddings"""

    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.config = config
        Classifier.__init__(self)

    def make(self):
        Classifier.make(self)
        self.num_layers = self.config.num_layers
        self.hidden_dim = self.config.hidden_dim
        if type(self.hidden_dim) is int:
            self.hidden_dim = [self.hidden_dim] * self.num_layers
        self.model_class = self.config.name

    def get_model(self):
        error("Attempted to access base neuralnet model getter.")
        return None

    # def save_model(self, model):
    #     with open(self.get_current_model_path(), "wb") as f:
    #         torch.save(model, f)

    # def load_model(self):
    #     path = self.get_current_model_path()
    #     if not exists(path):
    #         return None
    #     with open(path, "rb") as f:
    #         state_dict = torch.load(f)
    #     model = self.get_model()
    #     model.load_state_dict(state_dict)
    #     return model

    def update_best(self, current_loss, model):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_model = deepcopy(model.state_dict())

    def validate_model(self, val_data_loader, val_labels, model):
        # decide on model with respect to validation
        model.eval()
        val_loss_func = nn.CrossEntropyLoss()
        val_preds = []

        for val_index_labels in val_data_loader:
            val_batch_index, val_batch_labels = val_index_labels
            batch_val_preds = model.forward(val_batch_index)
            val_preds.append(batch_val_preds)

        val_preds = torch.cat(val_preds)
        val_loss = val_loss_func(val_preds, torch.tensor(val_labels))
        self.update_best(val_loss, model)
        return val_loss

    def train_model(self):
        """Training method for pytorch networks"""
        train_index, embeddings, train_labels, val_index, val_labels = self.train_index, self.embeddings, self.train_labels, self.val_index, self.val_labels
        if val_index is not None:
            val_set = NeuralNet.TorchDataset(val_index, train_labels)
            val_set = NeuralNet.TorchDataset(val_index, train_labels)
            val_data_loader = DataLoader(val_set, batch_size=self.batch_size)
        else:
            val_data_loader = None
        self.best_loss, self.best_model = 9999, None

        # setup vars
        model = self.get_model()
        optimizer = optim.Adam(model.parameters())
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        loss_func = nn.CrossEntropyLoss()

        # training loop
        train_set = NeuralNet.TorchDataset(train_index, train_labels)
        data_loader = DataLoader(train_set, batch_size=self.batch_size)
        model.train()
        for epoch in range(self.epochs):
            for batch_index, index_labels in enumerate(data_loader):

                index, labels = index_labels
                optimizer.zero_grad()

                preds = model.forward(index)
                loss = loss_func(preds, labels)
                loss.backward()
                optimizer.step()

                if val_index is not None:
                    val_msg = ", validation loss: {:.3f}".format(loss.item())
                else:
                    val_msg = ""
                debug("Epoch {}, batch {} / {}, training loss {:.3f}{}".format(epoch + 1, batch_index + 1, len(data_loader), loss.item(), val_msg))

            if val_index is not None:
                val_loss = self.validate_model(val_data_loader, val_labels, model)
                if lr_scheduler:
                    lr_scheduler.step(val_loss)
            else:
                self.update_best(loss, model)
                if lr_scheduler:
                    lr_scheduler.step(loss)

        # get best model
        model.load_state_dict(self.best_model)
        return model

    def test_model(self, model):
        test_set = NeuralNet.TorchDataset(self.test_index)
        data_loader = DataLoader(test_set, batch_size=self.batch_size)
        predictions = np.ndarray((0, self.num_labels), np.float32)
        self.model.eval()
        with torch.no_grad():
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

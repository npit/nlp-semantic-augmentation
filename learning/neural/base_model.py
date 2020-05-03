import torch
from torch.nn import functional as F
import pytorch_lightning as ptl
from utils import error

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

class BaseModel(ptl.LightningModule):
    """Base class for pytorch models, organized as a pytorch-lightning module
    
    This class and its derivatives must implement
    a) all functionality required by pytorch nn modules  (forward, etc.)
    b) all functionality required by pytorch-lightning modules (train_dataloader, configure_optimizers, etc.)
    """
    config = None
    name = "BASE_MODEL"

    def __init__(self, config, wrapper_name):
        """Model constructor"""
        self.config = config
        self.name = self.config.name
        self.wrapper_name = wrapper_name
        super(BaseModel, self).__init__()

    class Dataset:
        """Dataset class to construct the required dataloaders"""
        def __init__(self, data, labels=None):
            self.data = data
            self.labels = labels
            
        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            datum = self.data[index]
            if self.labels is not None:
                return datum, self.labels[index]
            return datum

    def configure_embedding(self):
        # incorporate embeddings in the neural architecture
        error("Attempted to access abstract embedding configuration function.")

    # high-level functions for NN ptl operations
    ########################################
    def train_model(self):
        """Training and validation function"""
        # also check https://pytorch-lightning.readthedocs.io/en/latest/fast_training.html
        self.configure_optimizers()
        # trainer = Trainer(val_check_interval=100)
        trainer = Trainer(min_epochs=1, max_epochs=self.config.train.epochs)
        trainer.fit(self)

    def test_model(self):
        """Testing function
        
        Since we are interested in the model predictions and make the evaluation outside ptl, this should never be required to run.
        """
        error("Attempted to invoke test_model form within a ptl module -- a test_model() should be run in the wrapper class instead that invokes forward().")
        # trainer = Trainer()
        # trainer.test(self)

    # low-level functions for NN ptl steps
    # ####################################
    # forward fn -- should be defined in a subclass
    def forward(self, x):
        """Forward pass function"""
        # forward pass
        error("Attempted to access abstract forward function")

    def get_data_from_index(self, index, data):
        """Retrieve a subset of embedding data"""
        return data[index]

    def should_do_validation(self):
        return len(self.val_index) > 0

    # def get_data(self, index):
    #     return self.embeddings[index]

    # dataset / dataloader utils
    def make_dataset_from_index(self, index, labels):
        # data = self.get_data(index)
        return BaseModel.Dataset(index, labels)

    def prepare_data(self):
        """Preparatory data actions and/or writing to disk"""
        pass

    def train_dataloader(self):
        """Preparatory actions for training data"""
        self.train_dataset = self.make_dataset_from_index(self.train_index, self.train_labels)
        return DataLoader(self.train_dataset, self.config.train.batch_size)
    def val_dataloader(self):
        """Preparatory transformation actions for validation data"""
        if self.should_do_validation():
            self.val_dataset = self.make_dataset_from_index(self.val_index, self.val_labels)
            return DataLoader(self.val_dataset, self.config.train.batch_size)

    def test_dataloader(self):
        """Preparatory transformation actions for test data"""
        # self.test_dataset = self.make_dataset_from_index(self.test_index, self.test_labels)
        self.test_dataset = self.make_dataset_from_index(self.test_index, None)
        return DataLoader(self.test_dataset, self.config.train.batch_size)

    # training
    def configure_optimizers(self):
        """Setup optimizers for training"""
        # optimizers
        if self.config.train.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=1e-3) 
        elif self.config.train.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=1e-3) 

    def training_step(self, batch, batch_idx):
        """Define a single training step"""
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    # validation
    def validation_step(self, batch, batch_idx):
        """Define a single validation step"""
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

        # error("Attempted to access abstract validation step function")
    def validation_epoch_end(self, outputs):
        """Define metric computation at validation epoch end"""
        if self.should_do_validation():
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        return {}

    def test_step(self, batch, batch_idx):
        """Define a single testing step"""
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        """Define metric computation at test epoch end"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        # error("Attempted to access abstract test epoch end function")
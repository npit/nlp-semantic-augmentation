from learning.labelled_learner import LabelledLearner
from learning.supervised_learner import SupervisedLearner
from learning.learner import Learner
from learning.neural.models import instantiator
import torch
from os.path import join, exists
import numpy as np
from utils import info, debug, warning

class DNN:
    """ class for all pytorch-lightning deep neural networks"""
    neural_model = None
    neural_model_class = None

    embeddings = None
    input_shape = None

    def __init__(self):
        """Constructor"""
        self.neural_model_class = instantiator.get_neural_model_class(self.config.name)

    def assign_embedding_data(self, model_instance):
        model_instance.embeddings = self.embeddings

    def assign_train_data(self, model_instance):
        """Transfer input indexes to the nn model"""

        self.assign_embedding_data(model_instance)
        model_instance.train_index = torch.LongTensor(self.train_index)
        model_instance.val_index = torch.LongTensor(self.val_index)

    def assign_test_data(self, model_instance):
        """Transfer test input indexes to the nn model"""
        self.assign_embedding_data(model_instance)
        model_instance.test_index = torch.LongTensor(self.test_index)

    def get_model_filename(self, model_index=None):
        """Get model base filename"""
        return self.neural_model_class.name + super().get_model_filename(model_index)

    def test_model(self, model_instance):
        """Testing function"""
        # debug(f"Setting eval mode")
        model_instance.eval()
        # debug(f"Assigning test data")
        self.assign_test_data(model_instance)
        # debug(f"Setting no-grad")
        with torch.no_grad():
            predictions = []
            # defer to model's forward function
            for input_batch in model_instance.test_dataloader():
                # debug(f"Testing batch {input_batch.shape}")
                batch_predictions = model_instance.make_predictions(input_batch)
                # debug(f"Got predictions: {batch_predictions.shape}")
                batch_predictions = self.process_predictions(batch_predictions)
                # batch_predictions = model_instance(input_batch)
                # account for possible batch padding TODO fix
                input_len = input_batch.numel() if input_batch.ndim == 0 else len(input_batch)
                batch_predictions = batch_predictions[:input_len]
                predictions.append(batch_predictions)
        return np.concatenate(predictions, axis=0)

    def process_predictions(self, preds):
        return preds.cpu()

    def save_model(self):
        """Model saving function for DNNs"""
        for idx, model in enumerate(self.models):
            self.model_index = idx
            path = self.get_current_model_path()
            info("Saving model to {}".format(path))
            torch.save(model.state_dict(), path)
            # save wrapper as well
            self.save_model_wrapper()

    def load_model_weights(self, path):
        state_dict = torch.load(path)
        return state_dict

    def get_input_shape_from_weights(self, state_dict):
        self.input_shape = state_dict['embedding_layer.weight'].shape

    def load_model(self):
        """Model loading function for DNNs"""
        path = self.get_current_model_path()
        if not exists(path):
            return False
        if not super().load_model_wrapper():
            info(f"Failed to load wrapper metadata for {self.name}")
            return False
        state_dict = self.load_model_weights(path)
        self.get_input_shape_from_weights(state_dict)
        # instantiate object
        self.build_model()
        # assign weights to the net
        try:
            self.neural_model.load_state_dict(state_dict)
        except RuntimeError as rte:
            warning(str(rte))
            warning("Weights loading failed -- aborting model loading.")
            self.clear_model_wrapper()
            return False
        return True

    def get_model(self):
        return self.neural_model

    def get_embedding_info(self):
        input_dim_info = self.embeddings if self.embeddings is not None else self.input_shape
        return input_dim_info

    def configure_model_after_inputs(self):
        if not self.config.retain_embedding_matrix:
            self.neural_model.update_embedding_layer(self.embeddings)

class GenericSupervisedDNN(DNN):
    """Generic class for deep neural networks with ground truth"""
    def __init__(self):
        """Constructor"""
        DNN.__init__(self)

    def assign_train_data(self, model_instance):
        """Transfer input indexes and ground truth to the nn model"""
        # call base DNN function to assign for instance indexes
        DNN.assign_train_data(self, model_instance)
        # also ground truth
        model_instance.assign_ground_truth(self.get_ground_truth())
        # model_instance.train_labels = self.train_labels
        # model_instance.val_labels = self.val_labels


class SupervisedDNN(GenericSupervisedDNN, SupervisedLearner):
    """Class for DNNs with non-numeric ground truth"""
    name = "supervised_dnn"

    def __init__(self, config):
        """Constructor"""
        self.config = config
        GenericSupervisedDNN.__init__(self)
        SupervisedLearner.__init__(self)

    def train_model(self):
        """Training function"""
        self.build_model()
        # defer to model's function
        self.assign_train_data(self.neural_model)
        self.neural_model.train_model()
        return self.neural_model

    def get_model(self):
        """Baseline model instantiation"""
        # instantiate the model
        self.neural_model = self.neural_model_class(self.config, self.get_ground_truth_info(), self.embeddings.shape[-1])
        self.neural_model.configure_embedding(len(self.embeddings), self.embeddings.shape[-1])

class LabelledDNN(GenericSupervisedDNN, LabelledLearner):
    name = "labelled_dnn"
    def __init__(self, config):
        self.config = config
        GenericSupervisedDNN.__init__(self)
        LabelledLearner.__init__(self)

    def load_model_weights(self, path):
        state_dict = super().load_model_weights(path)
        return state_dict

    def get_ground_truth_info(self):
        return self.get_output_dim()
    def get_output_dim(self):
        return self.num_labels

    def build_model(self):
        """Model building function for simple labelled DNNs"""
        self.num_labels = len(self.labels_info.labelset)

        self.neural_model = self.neural_model_class(self.config, self.get_embedding_info(),
            output_dim=self.get_output_dim(), working_folder=self.config.folders.results, model_name=self.get_model_filename())
        # print(self.neural_model)


    def train_model(self):
        """Training function"""
        self.build_model()
        # defer to model's function
        self.assign_train_data(self.neural_model)
        self.neural_model.train_model()
        return self.neural_model


class UnsupervisedDNN(DNN, Learner):
    """A supervised deep neural network"""
    name = "unsupervised_dnn"

    def __init__(self):
        DNN.__init__(self)
        Learner.__init__(self)

    def build_model(self):
        """Build the model"""
        self.neural_model = self.neural_model_class(self.config, self.get_embedding_info(), self.num_clusters)

    def train_model(self):
        """Training function"""
        # assign data
        self.assign_train_data()
        # defer to model's function
        self.model.train_model()

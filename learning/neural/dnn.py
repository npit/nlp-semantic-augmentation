from learning.labelled_learner import LabelledLearner
from learning.supervised_learner import SupervisedLearner
from learning.learner import Learner
from learning.neural.models import instantiator
import torch
from os.path import join
import numpy as np
from utils import info

class DNN:
    """ class for all pytorch-lightning deep neural networks"""
    neural_model = None
    neural_model_class = None

    def __init__(self):
        """Constructor"""
        self.neural_model_class = instantiator.get_neural_model_class(self.config.name)

    def assign_train_data(self, model_instance):
        """Transfer input indexes to the nn model"""
        model_instance.embeddings = self.embeddings

        model_instance.train_index = torch.LongTensor(self.train_index)
        model_instance.val_index = torch.LongTensor(self.val_index)

    def assign_test_data(self, model_instance):
        """Transfer test input indexes to the nn model"""
        model_instance.test_index = torch.LongTensor(self.test_index)

    def get_current_model_path(self):
        name = self.name + "_" + self.neural_model_class.name
        return self.validation.modify_suffix(
            join(self.results_folder, "models", "{}".format(name))) + ".model"

    def test_model(self, model_instance):
        """Testing function"""
        model_instance.eval()
        self.assign_test_data(model_instance)
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            predictions = []
            # defer to model's forward function
            for input_batch in model_instance.test_dataloader():
                batch_predictions = model_instance.make_predictions(input_batch)
                # batch_predictions = model_instance(input_batch)
                # account for possible batch padding TODO fix
                batch_predictions = batch_predictions[:len(input_batch)]
                predictions.append(batch_predictions)
        return np.concatenate(predictions, axis=0)

    def build_model(self):
        """Build the model"""
        self.neural_model = self.neural_model_class(self.config, self.embeddings, output_dim=self.get_output_dim(), working_folder=self.config.folders.results, model_name=self.get_model_instance_name())
        print(self.neural_model)

    def save_model(self):
        path = self.get_current_model_path()
        info("Saving model to {}".format(path))
        torch.save(self.neural_model.state_dict(), path)

    def load_model(self):
        path = self.get_current_model_path()
        # instantiate object
        self.build_model()
        # load weights
        self.neural_model.load_state_dict(torch.load(path))

    # def train_validate_dnn(self):
    #     """Training and validation function"""
    #     # also check https://pytorch-lightning.readthedocs.io/en/latest/fast_training.html
    #     self.model.configure_optimizers()
    #     trainer = Trainer()
    #     trainer.fit(self.model)

    # def test_dnn(self):
    #     """Testing function"""
    #     trainer = Trainer()
    #     trainer.test(self.model)

class SupervisedDNN(DNN, SupervisedLearner):
    """A supervised deep neural network"""
    name = "supervised_dnn"

    def __init__(self, config):
        """Constructor"""
        self.config = config
        DNN.__init__(self)
        SupervisedLearner.__init__(self)

    def assign_train_data(self, model_instance):
        """Transfer input indexes and ground truth to the nn model"""
        # base DNN for insance indexes
        super(SupervisedDNN, self).assign_train_data(model_instance)
        # also ground truth
        model_instance.assign_ground_truth(self.get_ground_truth())
        # model_instance.train_labels = self.train_labels
        # model_instance.val_labels = self.val_labels

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

class LabelledDNN(SupervisedDNN):
    def __init__(self):
        SupervisedDNN.__init(self)
    def get_ground_truth_info():
        return self.get_output_dim()
    def get_output_dim():
        return self.num_labels

class UnsupervisedDNN(DNN, Learner):
    """A supervised deep neural network"""
    name = "unsupervised_dnn"

    def __init__(self):
        DNN.__init__(self)
        Learner.__init__(self)

    def build_model(self):
        """Build the model"""
        self.neural_model = self.neural_model_class(self.config, self.embeddings, self.num_clusters)

    def train_model(self):
        """Training function"""
        # assign data
        self.assign_train_data()
        # defer to model's function
        self.model.train_model()

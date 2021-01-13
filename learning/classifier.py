import numpy as np
from sklearn.dummy import DummyClassifier as sk_Dummy
from sklearn.linear_model import LogisticRegression as sk_LogReg
from sklearn.naive_bayes import GaussianNB as sk_NaiveBayes
from sklearn.preprocessing import StandardScaler

from learning.labelled_learner import LabelledLearner
from utils import (error, ill_defined, one_hot, read_pickled, warning,
                   write_pickled)


class Classifier(LabelledLearner):

    def __init__(self):
        """Generic classifier constructor
        """
        LabelledLearner.__init__(self)

    def make(self):
        # make sure there exist enough labels
        LabelledLearner.make(self)
        error("Dataset supplied to classifier has only one label", ill_defined(self.num_labels, cannot_be=1))

    def is_supervised(self):
        """All classifiers require label information"""
        return True

    def get_model(self):
        return self.model


class SKLClassifier(Classifier):
    """Scikit-learn classifier"""
    args = {}

    def __init__(self):
        Classifier.__init__(self)

    def make(self):
        Classifier.make(self)

    # split train/val labels, do *not* convert to one-hot
    def prepare_labels(self, trainval_idx):
        train_idx, val_idx = trainval_idx
        if len(train_idx) > 0:
            train_labels = [self.train_labels[i] for i in train_idx]
        else:
            train_labels = np.empty((0,))
        if len(val_idx) > 0:
            val_labels = [self.train_labels[i] for i in val_idx]
        else:
            val_labels = np.empty((0,))
        return train_labels, val_labels

    def train_model(self):
        train_data = self.get_data_from_index(self.train_index, self.embeddings)
        train_labels = self.targets.get_slice(self.train_index)
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_data)
        self.model = self.model_class(**self.args)
        self.model.fit(train_data, np.asarray(train_labels).ravel())
        return (self.model, self.scaler)

    def load_model(self):
        self.model_loaded = super().load_model()
        if self.model_loaded:
            self.model, self.scaler = self.model
        return self.model_loaded

    def get_model(self):
        return (self.model, self.scaler)

    # evaluate a clustering
    def test_model(self, model):
        model, scaler = model
        test_data = self.get_data_from_index(self.test_index, self.embeddings)
        test_data = scaler.transform(test_data)
        predictions = model.predict_proba(test_data)
        # # convert back to one-hot
        # predictions = one_hot(predictions, self.num_labels, self.do_multilabel)
        return predictions



class NaiveBayes(SKLClassifier):
    name = "naive_bayes"

    def __init__(self, config):
        self.config = config
        self.model_class = sk_NaiveBayes
        SKLClassifier.__init__(self)

    def make(self):
        error("Cannot apply {} to multilabel data.".format(self.name), self.do_multilabel)
        SKLClassifier.make(self)


class Dummy(SKLClassifier):
    name = "dummy"

    def __init__(self, config):
        self.config = config
        self.model_class = sk_Dummy
        SKLClassifier.__init__(self)

    def make(self):
        error("Cannot apply {} to multilabel data.".format(self.name), self.do_multilabel)
        SKLClassifier.make(self)

class LogisticRegression(SKLClassifier):
    name = "logreg"

    def __init__(self, config):
        self.config = config
        self.model_class = sk_LogReg
        self.args = {"solver": "lbfgs", "max_iter": config.train.epochs, "verbose": 1}
        SKLClassifier.__init__(self)

    def make(self):
        error("Cannot apply {} to multilabel data.".format(self.name), self.do_multilabel)
        SKLClassifier.make(self)

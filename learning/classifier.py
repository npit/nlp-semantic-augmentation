from learning.learner import Learner
from utils import error, one_hot, ill_defined, warning, write_pickled, read_pickled
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier


class Classifier(Learner):

    def __init__(self):
        """Generic classifier constructor
        """
        Learner.__init__(self)

    def make(self):
        # make sure there exist enough labels
        Learner.make(self)
        error("Dataset supplied to classifier has only one label", ill_defined(self.num_labels, cannot_be=1))


class SKLClassifier(Classifier):
    """Scikit-learn classifier"""
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

    def train_model(self, train_data, train_labels, val_data, val_labels):
        model = self.model()
        model.fit(train_data, np.asarray(train_labels).ravel())
        return model

    # evaluate a clustering
    def test_model(self, test_data, model):
        predictions = model.predict(test_data)
        # convert back to one-hot
        predictions = one_hot(predictions, self.num_labels)
        return predictions

class NaiveBayes(SKLClassifier):
    name = "naive_bayes"

    def __init__(self, config):
        self.config = config
        self.model = GaussianNB
        SKLClassifier.__init__(self)

    def make(self):
        # if dataset.is_multilabel():
        #     error("Cannot apply {} to multilabel data.".format(self.name))
        warning("Add multilabel check")
        SKLClassifier.make(self)

class Dummy(SKLClassifier):
    name = "dummy"
    def __init__(self, config):
        self.config = config
        self.model = DummyClassifier
        SKLClassifier.__init__(self)

    def make(self):
        # if dataset.is_multilabel():
        #     error("Cannot apply {} to multilabel data.".format(self.name))
        warning("Add multilabel check")
        SKLClassifier.make(self)

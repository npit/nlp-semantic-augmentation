from learner import Learner
from utils import tictoc, write_pickled, info, error, read_pickled
import numpy as np


class Classifier(Learner):

    def __init__(self):
        """Generic classifier constructor
        """
        Learner.__init__(self)

    def make(self, representation, dataset):
        Learner.make(self, representation, dataset)

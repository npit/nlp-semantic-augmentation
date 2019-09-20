import numpy as np


class Text:
    name = "text"
    instances = None
    vocabulary = None
    def __init__(self, inst, vocab=None):
        self.instances = inst
        self.vocabulary = vocab

class Vectors:
    name = "vectors"
    instances = None
    elements_per_instance = None
    def __init__(self, vecs, epi=None):
        self.instances = vecs
        if epi is None:
            try:
                epi = [np.ones(len(x)) for x in self.instances]
            except:
                epi = [len(self.instances)]
        self.elements_per_instance = epi

    def get_instances(self):
        return self.instances

class Labels:
    name = "labels"
    instances = None
    def __init__(self, labels):
        self.instances = labels

class Indices:
    name = "indices"
    instances = None
    def __init__(self, indices):
        self.instances = indices

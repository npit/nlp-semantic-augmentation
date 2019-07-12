import numpy as np

class Text:
    name = "text"
    instances = None
    vocabulary = None
    def __init__(self, inst, vocab=None):
        self.instances = inst
        self.vocabulary = vocab
        pass

class Vectors:
    name = "vectors"
    instances = None
    elements_per_instance = None
    def __init__(self, vecs, epi=None):
        self.instances = vecs
        if epi is None:
            epi = [np.ones(len(x)) for x in self.instances]
        self.elements_per_instance = epi
    def get_instances(self):
        return self.instances

class Labels:
    name = "labels"
    instances = None
    def __init__(self, labels):
        self.instances = labels

import numpy as np

from bundle.bundle import DataPool
from bundle.datatypes import Numeric
from manip.manip import Manipulation
from utils import error, info, shapes_list, warning

"""Class that implements the combination of two chains"""


class Fusion(Manipulation):

    def __init__(self, config):
        self.config = config
        Manipulation.__init__(self)
        # self.num_inputs = len(config.inputs)

    def fuse(self):
        error("Attempted to call abstract fuse() from component {}".format(self.name))

    def produce_outputs(self):

        info("{self.name} fusion order:")
        for i, vecs in enumerate(self.vectors):
            output_vectors = self.fuse(vecs.instances)
            info(f"Applied {self.component_name} fusion {i+1}/{len(self.vectors)} with inputs: {vecs.instances.shape}, shape now: {output_vectors.shape}")
        self.outputs.set_vectors(Numeric(vecs=output_vectors))
        # set the first index
        self.outputs.set_indices(self.indices)


        # # fuse existing vectors
        # self.vectors = list(filter(lambda x: x is not None, self.vectors))
        # num_vector_collections = list(map(len, self.vectors))
        # error("Inconsistent number of vector collections to fuse: {}".format(num_vector_collections), len(set(num_vector_collections)) != 1)
        # error("Inconsistent number of vector collections to fuse: {}".format(num_vector_collections), len(set(num_vector_collections)) != 1)
        # self.vectors = list(zip(*self.vectors))
        # for v, vecs in enumerate(self.vectors):
        #     msg = "Fusing input collection {}/{} with shapes: {} to".format(v + 1, len(self.vectors), shapes_list(vecs))
        #     self.vectors[v] = self.fuse(vecs)
        #     info(msg + " {}".format(self.vectors[v].shape))

        # self.outputs.set_vectors(Numeric(vecs=self.vectors, epi=[np.ones(len(vecs), np.int32) * self.num_elements_per_instance for vecs in self.vectors]))

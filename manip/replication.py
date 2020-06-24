from bundle.datatypes import Vectors
import numpy as np
from utils import info, error

from manip.manip import Manipulation


class Replication(Manipulation):
    """Replicates input vector rows
    """
    name = "repl"
    replicate_times = None

    def __init__(self, config):
        Manipulation.__init__(self)
        self.replicate_times = config.manip.times

    def configure_name(self):
        Manipulation.configure_name(self)
        # add number of replications
        self.name += "_" + str(self.replicate_times)
        self.set_source_name(self.name)

    def run(self):
        self.process_component_inputs()
        if type(self.vectors) is tuple:
            self.vectors = list(self.vectors)
        for v, vec in enumerate(self.vectors):
            # replicate k rows: from MxN to Mx(N+k), reshape to N-dimensional
            msg = "Replicating input collection with shape: {} to".format(vec.shape)
            vec = np.reshape(np.tile(vec, (1, self.replicate_times)), (-1, vec.shape[-1]))
            info(msg + " {}".format(vec.shape))
            self.vectors[v] = vec
        self.outputs.set_vectors(Vectors(vecs=self.vectors, epi=[np.ones(len(vec), np.int32) * self.replicate_times for vec in self.vectors]))
        self.outputs.set_source_name(self.name)

    # def process_component_inputs(self):
    #     # input is a bundle
    #     Manipulation.process_component_inputs(self)
    #     self.vectors = self.vectors[0]
    #     error("Passed a bundle list to replicate.", len(self.vectors) != 1)


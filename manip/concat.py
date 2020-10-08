from manip.fusion import Fusion
from utils import shapes_list, error, data_summary, debug
import numpy as np

"""Vector concatenation manip
Only vectors are affected
"""
class Concatenation(Fusion):
    # column-wise concatenation
    name = "concat"
    axis = 1

    output = None

    def __init__(self, config):
        Fusion.__init__(self, config)

    def produce_outputs(self):
        self.outputs = np.concatenate(self.vectors, axis=1)
        # dim = self.vectors[0].shape[-1]
        # dtype = self.vectors[0].dtype
        # self.output = np.ndarray((0, dim), dtype=dtype)
        # for v in self.vectors:
        #     self.output = np.append(self.output, v, axis=1)
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

    def fuse(self, input_vectors):
        try:
            debug(f"Concatenating input chunck {input_vectors.shape}")
            if self.output is None:
                # output has the same number of instances as the input
                self.output = input_vectors
            else:
                self.output = np.concatenate([self.output, input_vectors], axis=1)
            return self.output
        except:
            msg = "Error during {} manip.".format(self.name)
            error(msg)
        return None

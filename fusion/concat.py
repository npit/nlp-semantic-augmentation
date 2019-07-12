from fusion.fusion import Fusion
from utils import shapes_list, error, data_summary
import numpy as np

"""Vector concatenation fusion
Only vectors are affected
"""
class Concatenation(Fusion):
    # column-wise concatenation
    name = "concat"
    axis = 1

    def __init__(self, config):
        Fusion.__init__(self, config)

    def fuse(self, input_list):
        try:
            return np.concatenate(input_list, axis=1)
        except:
            msg = "Error during {} fusion.".format(self.name)
            error(msg)
        return None

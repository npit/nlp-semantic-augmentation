from manip.fusion import Fusion
from utils import shapes_list, error, data_summary, debug, info
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
        info(f"Manipulating {self.name} inputs: {shapes_list(self.vectors)}")
        self.outputs = np.concatenate(self.vectors, axis=1)
        info(f"Produced {self.name} outputs: {self.outputs.shape}")

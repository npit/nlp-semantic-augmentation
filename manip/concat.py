from manip.fusion import Fusion
from utils import shapes_list, error, data_summary, debug, info
from bundle.datatypes import *
from bundle.datausages import *
import numpy as np

"""
Vector concatenation manipulation component
"""
class Concatenation(Fusion):
    # column-wise concatenation
    name = "concat"
    axis = 1

    output = None

    def __init__(self, config):
        Fusion.__init__(self, config)

    def produce_outputs(self):
        data = [x.data for x in self.input_dps]
        data_classes = [get_data_class(x.data) for x in self.input_dps]
        if not all(issubclass(x, Numeric) for x in data_classes):
            error(f"{self.name} requires numeric-only input data, but was fed: {data_classes}")
        insts = [x.instances for x in data]
        for d in self.input_dps:
            info(f"{d} : {d.data.instances.shape}")
        info(f"Manipulating {self.name} inputs: {shapes_list(insts)}")
        self.outputs = np.concatenate(insts, axis=1)
        info(f"Produced {self.name} outputs: {self.outputs.shape}")

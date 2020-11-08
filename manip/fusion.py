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

    def get_component_inputs(self):
        super().get_component_inputs()
        error(f"Specified {self.name} component but number of input numerics found is {len(self.inputs)} -- need at least 2", len(self.inputs) < 2)
        # same indexes required
        if not all(x.equals(self.indices[0]) for x in self.indices):
            for x in self.indices:
                warning(str(x))
            error(f"{self.name} inputs are annotated with dfferent indices: {self.indices}")
        self.indices = self.indices[0]
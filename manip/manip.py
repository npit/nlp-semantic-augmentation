"""Vector manipulation"""
from bundle.bundle import DataPool
from bundle.datatypes import Numeric
from bundle.datausages import Indices, DataPack
from component.component import Component
from utils import error, info
import numpy as np


class Manipulation(Component):
    component_name = "manipulation"

    produces= (Numeric,)
    consumes= (Numeric,)

    def __init__(self):
        pass

    def configure_name(self):
        if type(self.source_name) in [list, tuple]:
            self.source_name = "_".join(self.source_name)
        self.name = "{}_{}".format(self.source_name, self.name)
        Component.configure_name(self, self.name)

    def get_component_inputs(self):
        self.vectors = []
        indices = []
        numerics = self.data_pool.request_data(Numeric, usage=Indices, usage_matching="subset", client=self.name, must_be_single=False)
        error(f"Specified {self.name} component but number of input numerics found is {len(numerics)} -- need at least 2", len(numerics) < 2)
        info(f"Manipulating via {self.name} data:")
        for i, n in enumerate(numerics):
            info(f"{i+1}: {n}")
            self.vectors.append(n.data.instances)
            ind = n.get_usage(Indices)
            indices.append(ind)
        if not all(x.equals(indices[0]) for x in indices):
            error(f"{self.name} inputs are annotated with dfferent indices: {ind}")

        self.indices = indices[0]

    def set_component_outputs(self):
        dp = DataPack(Numeric(self.outputs), self.indices)
        self.data_pool.add_data_packs([dp], self.name)

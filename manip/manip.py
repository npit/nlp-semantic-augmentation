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

    outputs = None

    def set_serialization_params(self):
        pass

    def configure_name(self):
        if type(self.source_name) in [list, tuple]:
            self.source_name = "_".join(self.source_name)
            self.name = "{}_{}".format(self.source_name, self.name)
        else:
            self.name = self.name
        Component.configure_name(self, self.name)

    def load_model_from_disk(self):
        return True

    def set_component_outputs(self):
        dp = DataPack(Numeric(self.outputs), self.indices)
        self.data_pool.add_data_packs([dp], self.name)

    def get_component_inputs(self):
        self.inputs = []
        self.indices = []
        # inputs = self.data_pool.request_data(Numeric, usage=Indices, usage_matching="subset", client=self.name, must_be_single=False)
        self.input_dps = self.data_pool.request_data(None, usage=Indices, usage_matching="subset", client=self.name, must_be_single=False)

        info(f"Manipulating via {self.name} data:")
        for i, n in enumerate(self.input_dps):
            info(f"{i+1}: {n}")
            self.inputs.append(n.data.instances)
            ind = n.get_usage(Indices)
            self.indices.append(ind)
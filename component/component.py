from bundle.bundle import DataPool
from bundle.bundle import Consumes, Produces

from utils import as_list, error


"""Abstract class representing a computation pipeline component
"""


class Component:
    config = None
    # IO data pool
    data_pool = None
    # variables to hold inputs & output types
    produces = None
    consumes = None

    component_name = None
    # required input  from other chains
    required_finished_chains = []

    def get_consumption(self, chain_name):
        self.consumes = as_list(self.consumes) if self.consumes is not None else []
        res = []
        for pr in self.consumes:
            try:
                dtype, usage = pr
            except ValueError:
                dtype, usage = pr, None
            res.append(Consumes(dtype, usage, self.get_name(), chain_name))
        return res
    def get_production(self, chain_name):
        self.produces = as_list(self.produces) if self.produces is not None else []
        res = []
        for pr in self.produces:
            try:
                dtype, usage = pr
            except ValueError:
                dtype, usage = pr, None
            res.append(Produces(dtype, usage, self.get_name(), chain_name))
        return res


    def get_component_name(self):
        return self.component_name

    def get_full_name(self):
        return "({}|{})".format(self.get_component_name(), self.get_name())

    def get_name(self):
        return self.component_name

    def get_required_finished_chains(self):
        return self.required_finished_chains

    def __init__(self):
        """Constructor"""
        pass

    def configure_name(self, name=None):
        # set configured name to the output bundle
        # if name is None:
        #     self.set_source_name(self.component_name)
        # else:
        #     self.set_source_name(name)
        pass

    def assign_data_pool(self, data_pool):
        self.data_pool = data_pool

    def get_outputs(self):
        """Outputs getter"""
        return self.outputs

    def __str__(self):
        return self.get_full_name()

    def run(self):
        """Component runner function"""
        error(f"Attempted to execute abstract component run function for {self.component_name}")

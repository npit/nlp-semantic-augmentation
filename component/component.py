from bundle.bundle import Bundle
from utils import as_list, error


"""Abstract class representing a computation pipeline component
"""


class Component:
    config = None
    # variables to hold inputs & outputs
    inputs = None
    outputs = None
    # variables to hold inputs & output types
    produces = None
    consumes = None

    component_name = None
    # required input  from other chains
    required_finished_chains = []

    def add_output_demand(self, chain_name, component_name):
        self.outputs.set_demand(chain_name, component_name)

    def get_component_name(self):
        return self.component_name

    def get_full_name(self):
        return "({}|{})".format(self.get_component_name(), self.get_name())

    def get_name(self):
        return self.component_name

    def get_required_finished_chains(self):
        return self.required_finished_chains

    def __init__(self, produces=None, consumes=None):
        """Constructor"""
        self.produces = as_list(produces)
        self.consumes = as_list(consumes)
        # instantiate output bundle
        self.outputs = Bundle()

    def configure_name(self, name=None):
        # set configured name to the output bundle
        if name is None:
            self.set_source_name(self.component_name)
        else:
            self.set_source_name(name)

    def set_source_name(self, name):
        self.outputs.set_source_name(name)

    def load_inputs(self, data):
        self.inputs = data

    def get_outputs(self):
        """Outputs getter"""
        return self.outputs

    def __str__(self):
        return self.get_full_name()

    def run(self):
        """Component runner function"""
        error(f"Attempted to execute abstract component run function for {self.component_name}")

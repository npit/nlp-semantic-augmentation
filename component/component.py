from bundle.bundle import Bundle
from utils import error
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
    can_be_final = False # denotes wether it is a valid chain end
    required_finished_chains = []

    def get_component_name(self):
        return self.component_name

    def get_full_name(self):
        return "({}|{})".format(self.get_component_name(), self.get_name())

    def get_name(self):
        return self.name

    def get_required_finished_chains(self):
        return self.required_finished_chains

    def __init__(self, produces=None, consumes=None):
        """Constructor"""
        self.produces = produces
        self.consumes = consumes
        # instantiate output bundle
        self.outputs = Bundle()
        pass

    def configure_name(self):
        # set configured name to the output bundle
        self.outputs.set_source_name(self.name)
        pass

    def ready(self, available_output_names):
        return all(x in available_output_names for x in self.required_finished_chains)

    def load_inputs(self, data):
        self.inputs = data

    def get_outputs(self):
        """Outputs getter"""
        return self.outputs

    def __str__(self):
        return "{} - {}".format(self.component_name, self.get_name())

    def run(self):
        """Component runner function"""
        error("Attempted to execute abstract component run function for {}".format(self.name))

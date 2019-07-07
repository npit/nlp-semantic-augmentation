from utils import error
"""Abstract class representing a computation pipeline component
"""


class Component:
    config = None
    inputs = None
    outputs = None
    component_name = None
    required_finished_chains = []

    def get_component_name(self):
        return self.component_name

    def get_name(self):
        return self.name

    def get_required_finished_chains(self):
        return self.required_finished_chains

    def __init__(self):
        """Constructor"""
        pass

    def ready(self, available_outputs):
        return all(x in available_outputs for x in self.required_finished_chains)

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

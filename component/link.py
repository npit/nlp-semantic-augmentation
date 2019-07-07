from component.component import Component
from utils import error

class Link(Component):
    component_name = "link"

    def __init__(self, config):
        # no subclass for this
        self.name = self.component_name
        self.required_finished_chains = config.link
        if type(self.required_finished_chains) is list:
            pass
        elif type(self.required_finished_chains) is str:
            self.required_finished_chains = [config.link]
        elif type(self.required_finished_chains) is tuple:
            self.required_finished_chains = list(config.link)
        else:
            error("Failed {} component parameter: {}".format(self.component_name, config.link))

    def run(self):
        # carry over inputs
        self.outputs = self.inputs

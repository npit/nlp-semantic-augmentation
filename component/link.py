from component.component import Component

class Link(Component):
    component_name = "link"

    def __init__(self, config):
        # no subclass for this
        self.name = self.component_name
        self.required_finished_chains = config.link

    def run(self):
        # carry over inputs
        self.outputs = self.inputs

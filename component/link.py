from component.component import Component

class Link(Component):
    name = "link"

    def __init__(self, config):
        self.required_finished_chains = config.link

    def run(self):
        # carry over inputs
        self.outputs = self.inputs

from component.component import Component
from component import instantiator
from utils import info, debug, error


class Chain(Component):
    components = None
    input_requirements = None
    name = None

    def get_components(self):
        return self.components

    def get_name(self):
        return self.name

    def __init__(self, name, fields, configs):
        """Constructor"""

        self.components = []
        info("Creating chain: [{}]".format(name))
        self.name = name
        for idx, (component_name, component_params) in enumerate(zip(fields, configs)):
            component = instantiator.create(component_name, component_params)
            self.components.append(component)
            debug("{}: {}".format(idx + 1, str(self.components[-1])))
        self.num_components = len(self.components)
        info("Created chain with {} components.".format(self.num_components))

    def run(self, available_chain_outputs):
        info("Running chain [{}]".format(self.name))
        info("-------------------")
        data = None
        for c, component in enumerate(self.components):
            info("Running component {}/{} : {}".format(c + 1, self.num_components, component))
            if component.required_finished_chains:
                if data is None:
                    data = [available_chain_outputs[x] for x in component.required_finished_chains]
                    if len(data) == 1:
                        data = data[0]
                else:
                    error("WHops-Required finished chains with existing inputs!")
            component.load_inputs(data)
            component.run()
            data = component.get_outputs()
            # mark current output as chain's output
            self.outputs = data

    def __str__(self):
        return self.get_name()

    def ready(self, chain_outputs=None):
        # a chain is ready if its first element is ready
        return self.components[0].ready(chain_outputs)

    def get_required_finished_chains(self):
        return self.components[0].required_finished_chains

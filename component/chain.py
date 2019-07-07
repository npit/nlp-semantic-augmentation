from component.component import Component
from component import instantiator
from utils import info, debug, error, data_summary


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
        # info("Creating chain: [{}]".format(name))
        # info("-------------------")
        self.num_components = len(fields)
        self.name = name
        for idx, (component_name, component_params) in enumerate(zip(fields, configs)):
            component = instantiator.create(component_name, component_params)
            self.components.append(component)
            debug("Created chain {} component {}/{}: {}".format(name, idx + 1, self.num_components, str(self.components[-1])))
        info("Created chain with {} components.".format(self.num_components))

    def run(self):
        info("Running chain [{}]".format(self.name))
        info("-------------------")
        data_bundle = None
        for c, component in enumerate(self.components):
            info("Running component {}/{} : type: {} - name: {}".format(c + 1, self.num_components, component.get_component_name(), component.get_name()))
            if component.get_required_finished_chains():
                # the component requires an input from another chain
                error("Chain [{}] requires input(s), but none are available.".format(self.get_name()), self.inputs is None)
                data_bundle = self.inputs.get_bundles(chain_names=component.get_required_finished_chains())
            if data_bundle is not None:
                data_bundle.summarize_content("Passing bundle(s) to component [{}]".format(component.get_name()))
            component.load_inputs(data_bundle)
            component.run()
            # data_bundle = {"data": component.get_outputs(), "name": component.get_name(), "component": component.get_component_name()}
            data_bundle = component.get_outputs()
            # data_summary(data_bundle, "output of component {}".format(component.get_name()))
            # debug("Component [{}] yielded an output of {}".format(component.get_name(), data))
            # mark current output as chain's output
            self.outputs = data_bundle

    def __str__(self):
        return self.get_name()

    def ready(self, chain_outputs=None):
        # a chain is ready if its first element is ready
        return self.components[0].ready(chain_outputs)

    def get_required_finished_chains(self):
        return self.components[0].required_finished_chains

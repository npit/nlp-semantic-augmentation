import defs
from component import instantiator
from component.component import Component
from utils import debug, error, info
from bundle.bundle import DataPool


class Chain(Component):
    """Class to represent an ordered sequence of components"""
    components = None
    input_requirements = None
    name = None

    def get_components(self):
        return self.components

    def get_name(self):
        return self.name

    def __init__(self, name, fields, configs):
        """Constructor for the chain"""
        self.components = []
        # parse potential chain inputs
        if defs.alias.link in fields:
            idx = fields.index(defs.alias.link)
            link_component = configs[idx]
            self.required_finished_chains = link_component.get_links()
            del fields[idx]
            del configs[idx]

        self.num_components = len(fields)
        self.name = name
        for idx, (component_name, component_params) in enumerate(zip(fields, configs)):
            component = instantiator.create(component_name, component_params)
            self.components.append(component)
            debug("Created chain {} component {}/{}: {}".format(name, idx + 1, self.num_components, str(self.components[-1])))
        # info("Created chain with {} components.".format(self.num_components))

    # def backtrack_output_demand(self, requesting_chain_name, requesting_component_name, consumes, data_pool):
    #     """Marks backwards the requirements of the chain's output"""
    #     debug("Backtracking needs of chain {} (first:{}) to chain {}".format(chain_name, component_name, self.name))
    #     cons = [x for x in consumes]
    #     # start with the last component
    #     for comp in reversed(self.components):

    #         data_pool.set_demand(chain_name, component_name)
    #     self.outputs.set_demand(chain_name, component_name)

    #         comp.add_output_demand(chain_name, component_name)
    #         overlap = [x for x in comp.produces if x in consumes]
    #         cons = [c for c in cons if c not in overlap]
    #         debug("Component {} covers {}need {}".format(comp.get_full_name(), "final" if not cons else "", overlap))
    #         if not cons:
    #             break

    def run(self, data_pool):
        """Runs the chain"""
        info("-------------------")
        info("{} chain [{}]".format("Running", self.name))
        info("-------------------")

        data_pool.on_chain_start(self.get_name())
        # iterate the chain components
        for c, component in enumerate(self.components):
            info("||| Running component {}/{} : type: {} - name: {}".format(c + 1, self.num_components, component.get_component_name(), component.get_name()))
            component.assign_data_pool(data_pool)
            component.run()
            data_pool.on_component_completion(self.get_name(), component.get_name())
            data_pool.clear_feeders()
            data_pool.add_feeders(None, component.get_name())
        data_pool.on_chain_completion(self.get_name())

    def configure_component_names(self):
        for c, component in enumerate(self.components):
            component.configure_name()
            debug("Named chain {} component #{}/{} : {}".format(self.get_name(), c + 1, len(self.components), component.get_full_name()))

    def __str__(self):
        return self.get_name()

    def ready(self, chain_output_names=None):
        # a chain is ready if its first element is ready
        return all(x in chain_output_names for x in self.required_finished_chains)

import defs
from component import instantiator
from component.component import Component
from utils import debug, error, info


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

    def backtrack_output_demand(self, chain_name, component_name, consumes):
        """Marks backwards the requirements of the chain's output"""
        debug("Backtracking needs of chain {} (first:{}) to chain {}".format(chain_name, component_name, self.name))
        cons = [x for x in consumes]
        for comp in reversed(self.components):
            # comp.add_output_demand(chain_name, component_name)
            comp.add_output_demand(chain_name, component_name)
            overlap = [x for x in comp.produces if x in consumes]
            cons = [c for c in cons if c not in overlap]
            debug("Component {} covers {}need {}".format(comp.get_full_name(), "final" if not cons else "", overlap))
            if not cons:
                break

    def run(self):
        """Runs the chain"""
        info("-------------------")
        info("{} chain [{}]".format("Running", self.name))
        info("-------------------")
        # chain inputs
        data_bundle = self.inputs 

        # if self.get_required_finished_chains():
        #     # the chain requires the output of other chains
        #     error("Chain [{}] requires input(s), but none are available.".format(self.get_name()), self.inputs is None)
        #     data_bundle = self.inputs.get_request_bundlelist(self.get_name())
        #     print(type(data_bundle))

        # iterate the chain components
        for c, component in enumerate(self.components):
            info("||| Running component {}/{} : type: {} - name: {}".format(c + 1, self.num_components, component.get_component_name(), component.get_name()))
            if data_bundle is not None:
                data_bundle.set_active_linkage(self.get_name())
                info(f"Active data bundle linkage: {data_bundle.active_linkage}")
                # data_bundle = data_bundle.find_head_with_linkage(data_bundle.active_linkage)
                # data_bundle.print_linkage_sequence(data_bundle.active_linkage)
                data_bundle = data_bundle.find_linkage_list_head(data_bundle.active_linkage)
                component.load_inputs(data_bundle)
            # if data_bundle is not None:
            #     data_bundle.summarize_content("Passing bundle(s) to component [{}]".format(component.get_name()))
            component.run()
            # check if input needs deletion now
            if data_bundle is not None:
                data_bundle.clear_data(self.get_name(), component.get_name())
            # update current component and chain output
            data_bundle = component.get_outputs()
            self.outputs = data_bundle
        # chain done - set the source chain name
        self.outputs.set_chain_name(self.name)

    def configure_component_names(self):
        for c, component in enumerate(self.components):
            component.configure_name()
            debug("Named chain {} component #{}/{} : {}".format(self.get_name(), c + 1, len(self.components), component.get_full_name()))

    def __str__(self):
        return self.get_name()

    def ready(self, chain_output_names=None):
        # a chain is ready if its first element is ready
        return all(x in chain_output_names for x in self.required_finished_chains)

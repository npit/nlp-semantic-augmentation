from bundle.bundle import Bundle
from utils import debug, error, info, warning


class Pipeline:
    """A collection of execution chains"""
    chains = None

    def __init__(self):
        """Constructor"""
        self.chains = {}

    def visualize(self):
        """Visualization function"""
        max_index = "{:>3s}"
        max_chainname = "{:" + str(max([len(x) for x in self.chains] + [len("chain name")])) + "s}"
        max_inputs = str(max([len(", ".join(x.get_required_finished_chains())) for x in self.chains.values()] + [len("req. inputs")]))
        max_inputs = "" if max_inputs == "0" else max_inputs
        max_inputs = "{:" + max_inputs + "s}"
        header = max_index + ") " + max_chainname + " | " + max_inputs + " | {}"
        hdr = header.format("-", "chain name", "req. inputs", "components")
        info(hdr)
        info("-" * len(hdr))

        for idx, (chain_name, chain) in enumerate(self.chains.items()):
            required_inputs = chain.get_required_finished_chains()
            if required_inputs:
                required_inputs = ", ".join(required_inputs)
            else:
                required_inputs = ""
            comps = chain.get_components()
            names = ["{}".format(x.get_full_name()) for x in comps]
            names = " -> ".join(names)
            info(header.format(str(idx + 1), chain_name, required_inputs, names))

    def get_next_configurable_chain(self, run_pool, chain_outputs):
        """Get the next runnable chain wrt the current run pool and the available completed chain outputs"""
        chain = self.chains[run_pool.pop(0)]
        required_input_chains = chain.get_required_finished_chains()
        # check if the chain requires inputs from other chains
        if required_input_chains:
            undefined_inputs = [x for x in required_input_chains if x not in self.chains]
            if undefined_inputs:
                error(f"Chain {chain.get_name()} requires an input from the non-existing chain(s): {undefined_inputs}")
            completed_chain_names = chain_outputs.get_chain_name(full_search=True)
            if not chain.ready(completed_chain_names):
                run_pool.append(chain.get_name())
                debug("Postponing chain {}".format(chain.get_name()))
                return None, None
        return chain, required_input_chains

    def configure_names(self):
        """Performs a simulated execution of member chains, to configure names of each component
        and initialize input / output bundles
        """
        info("Configuring pipeline.")
        completed_chain_outputs = None
        run_pool = list(self.chains.keys())
        while run_pool:
            chain, required_input_chains = self.get_next_configurable_chain(run_pool, completed_chain_outputs)
            if chain is None:
                continue

            data_bundle = None
            chain.load_inputs(completed_chain_outputs)
            if required_input_chains:
                # the first component requires an input from another chain -- check if it's loaded
                error("Chain [{}] requires input(s), but none are available.".format(chain.get_name()), chain.inputs is None)
                data_bundle = chain.inputs.make_request_bundlelist_by_chain_names(chain_names=required_input_chains, client_name=chain.get_name())
            for c, component in enumerate(chain.components):
                component.load_inputs(data_bundle)
                # configure names
                component.configure_name()
                # configure output dependencies
                if c > 0:
                    chain.components[c - 1].add_output_demand(chain.name, component.get_name())
                data_bundle = component.get_outputs()
                chain.outputs = data_bundle

            # inform chains feeding into the current one that their output is demanded
            if required_input_chains:
                first_component = chain.components[0].get_name()
                first_component_consumption = chain.components[0].consumes
                for ch in required_input_chains:
                    self.chains[ch].backtrack_output_demand(chain.get_name(), first_component, first_component_consumption)

            # update the current completed chain outputs
            output_bundle = chain.get_outputs()
            output_bundle.chain_name = chain.get_name()
            if completed_chain_outputs is None:
                completed_chain_outputs = output_bundle
            else:
                completed_chain_outputs.add_bundle(output_bundle)
        info("Pipeline configuration complete!")
        Bundle.print_linkages(completed_chain_outputs)
        for chain_name in self.chains:
            Bundle.print_linkages(completed_chain_outputs, linkage_name=chain_name)


    def run(self):
        """Executes the pipeline"""
        info("================")
        info("Running pipeline.")
        info("----------------")
        self.sanity_check()
        self.visualize()
        # chain_outputs = {ch: None for ch in self.chains}
        completed_chain_outputs = None
        completed_chain_names = []
        # poll a list of chains to run
        run_pool = list(self.chains.keys())
        while run_pool:
            chain = self.chains[run_pool.pop(0)]
            # check if the chain requires inputs from other chains
            if not chain.ready(completed_chain_names):
                debug("Delaying execution of chain {} since the required chain output {} is not available in the current ones: {}".format(chain.get_name(), str(chain.get_required_finished_chains()), completed_chain_names))
                run_pool.append(chain.get_name())
                continue
            # pass the entire finished chain output -- required content will be filtered at the chain start
            # chain.load_inputs(completed_chain_outputs)
            chain.run()

            # assign outputs
            if completed_chain_outputs is None:
                # we only need to assign the first output bundle
                # the rest is handled via the linkage mechanism
                completed_chain_outputs = chain.get_outputs()
            completed_chain_names.append(chain.get_name())

            # info(f"Default linkage after completion of chain {chain.get_name()}")
            # Bundle.print_linkages(completed_chain_outputs)

    def add_chain(self, chain):
        """Add a chain to the pipeline"""
        chain_name = chain.get_name()
        error("Duplicate chain: {}".format(chain_name), chain_name in self.chains)
        self.chains[chain_name] = chain

    def sanity_check(self):
        """Perform sanity-checking actions for  the entire pipeline"""
        all_required_inputs = set()
        # check input requirements are satisfiable
        for chain_name, chain in self.chains.items():
            for req_out in chain.get_required_finished_chains():
                if req_out not in self.chains:
                    error("Chain {} requires an output of a non-existent chain: {}".format(chain_name, req_out))
                all_required_inputs.add(req_out)

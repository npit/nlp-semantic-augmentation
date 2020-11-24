from bundle.bundle import DataPool
from utils import debug, error, info, warning


class Pipeline:
    """A collection of execution chains"""
    chains = None

    def __init__(self):
        """Constructor"""
        self.chains = {}
        self.data_pool = DataPool()

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

    def get_next_configurable_chain(self, run_pool, completed_chain_names):
        """Get the next runnable chain wrt the current run pool and the available completed chain outputs"""
        chain = self.chains[run_pool.pop(0)]
        required_input_chains = chain.get_required_finished_chains()
        # check if the chain requires inputs from other chains
        if required_input_chains:
            undefined_inputs = [x for x in required_input_chains if x not in self.chains]
            if undefined_inputs:
                error(f"Chain {chain.get_name()} requires an input from the non-existing chain(s): {undefined_inputs}")
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
        # presistent output data pool
        run_pool = list(self.chains.keys())
        completed_chains = []

        input_identifier = ""
        while run_pool:
            chain, required_input_chains = self.get_next_configurable_chain(run_pool, completed_chains)
            if chain is None:
                continue
            # make all finished chains available as potential inputs
            chain.assign_data_pool(self.data_pool)
            self.data_pool.on_chain_start(chain.get_name())
            if required_input_chains:
                # log the chain-level data demand
                # data_pool.log_data_request(chain.get_name(), required_input_chains)
                input_identifier = self.data_pool.get_input_identifier(chain.components[0].get_consumption(chain.get_name()), required_input_chains)

            debug(f"Configuring chain [{chain.get_name()}]")
            # run chain components
            for c, component in enumerate(chain.components):
                # # get appropriate head bundle
                component.assign_data_pool(self.data_pool)
                # configure names
                component.source_name = input_identifier
                component.configure_name()
                self.data_pool.log_data_consumption(component.get_consumption(chain.name))
                self.data_pool.log_data_production(component.get_production(chain.name))
                previous_component_name = component.get_name()
            completed_chains.append(chain.get_name())

            # for component in reversed(chain.components):
            #     # log data needs for the component
            #     chain_consumption.append(component.get_consumption())
            #     data_pool.log_data_request(component.get_consumption())
            #     # configure output dependencies for preceeding components
            #     if c > 0:
            #         previous_comp = chain.components[c - 1]
            #         # log the component-level data demand
            #         data_pool.log_data_request(previous_comp.get_name(), chain.get_name(), component.get_name())

            # # inform chains feeding into the current one that their output is demanded
            # if required_input_chains:
            #     first_component = chain.components[0].get_name()
            #     first_component_consumption = chain.components[0].consumes
            #     for ch in required_input_chains:
            #         self.chains[ch].backtrack_output_demand(chain.get_name(), first_component, first_component_consumption, data_pool)

        info("Pipeline configuration complete!")

    def configure_outputs(self):
        """Configure pipeline outputs"""
        # unclear what output should be
        # for now just return learner outputs
        for chain in self.chains.values():
            # if chain.get_components()
            # pass
            pass

    def load_models(self, failure_is_fatal=True):
        """Load the model from each component"""
        info("Preloading the models for each component")
        for chain in self.chains.values():
            for comp in chain.get_components():
                comp.attempt_load_model_from_disk(failure_is_fatal=failure_is_fatal)

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

            # set input chains as the feeders
            self.data_pool.clear_feeders()
            self.data_pool.add_feeders(chain.get_required_finished_chains(), None)
            
            # pass the entire finished chain output -- required content will be filtered at the chain start
            chain.run(self.data_pool)

            # assign outputs
            # if completed_chain_outputs is None:
            #     # we only need to assign the first output bundle
            #     # the rest is handled via the linkage mechanism
            #     completed_chain_outputs = chain.get_outputs()

            completed_chain_names.append(chain.get_name())

            # info(f"Default linkage after completion of chain {chain.get_name()}")
            # Bundle.print_linkages(completed_chain_outputs)
        outputs = self.data_pool.get_outputs()
        return outputs


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

    def setup_triggers():
        pss
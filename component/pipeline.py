from component.bundle import BundleList
from utils import error, info, warning, debug

class Pipeline:
    chains = None

    def __init__(self):
        """Constructor"""
        self.chains = {}

    def run(self):
        info("================")
        info("Running pipeline.")
        info("----------------")
        self.sanity_check()
        # chain_outputs = {ch: None for ch in self.chains}
        chain_outputs = BundleList()
        # store existing chain outputs to handle dependencies
        warning("Use a single dataset reading source -- for e.g. multiple representations, use a single dataset reading pipeline, and feed its output to multiple new ones")
        # poll a list of chains to run
        run_pool = list(self.chains.keys())
        while run_pool:
            chain = self.chains[run_pool.pop(0)]
            # check if the chain requires inputs from other chains
            if not chain.ready(chain_outputs.get_chain_names()):
                debug("Delaying execution of chain {} since the required chain output {} is not available in the current ones: {}".format(chain.get_name(), str(chain.get_required_finished_chains()), chain_outputs.get_names()))
                run_pool.append(chain.get_name())
                continue
            # pass the entire finished chain output -- required content will be filtered at the chain start
            chain.load_inputs(chain_outputs)
            chain.run()
            chain_outputs.add_bundle(chain.get_outputs(), chain_name=chain.get_name())
            # chain_outputs[chain.get_name()] = chain.get_outputs()

    def add_chain(self, chain):
        chain_name = chain.get_name()
        error("Duplicate chain: {}".format(chain_name), chain_name in self.chains)
        self.chains[chain_name] = chain

    def sanity_check(self):
        # check input requirements are satisfiable
        for chain_name, chain in self.chains.items():
            first_component = chain.get_components()[0]
            for req_out in first_component.get_required_finished_chains():
                if req_out not in self.chains:
                    error("First component [{}] of chain {} requires an output of a non-existent chain: {}".format(
                        first_component.get_name(), chain_name, req_out))

from utils import error, info, warning

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
        chain_outputs = {ch: None for ch in self.chains}
        # store existing chain outputs to handle dependencies
        warning("Use a single dataset reading source -- for e.g. multiple representations, use a single dataset reading pipeline, and feed its output to multiple new ones")
        # poll a list of chains to run
        run_pool = list(self.chains.keys())
        while run_pool:
            chain = self.chains[run_pool.pop(0)]
            if not chain.ready(chain_outputs):
                run_pool.append(chain)
                continue
            required_chain_outputs = chain.get_required_finished_chains()
            chain.run(chain_outputs)
            chain_outputs[chain.get_name()] = chain.get_outputs()

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
                    error("Component #0: {} of chain {} requires an output for a non-existent chain: {}".format(
                        first_component.get_name(), chain_name, req_out))

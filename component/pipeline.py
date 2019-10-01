from bundle.bundle import BundleList
from utils import error, info, warning, debug

class Pipeline:
    chains = None

    def __init__(self):
        """Constructor"""
        self.chains = {}

    def visualize(self):
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
            names =  " -> ".join(names)
            info(header.format(str(idx + 1), chain_name, required_inputs, names))

    def configure_names(self):
        chain_outputs = BundleList()
        run_pool = list(self.chains.keys())
        while run_pool:
            chain = self.chains[run_pool.pop(0)]
            required_input_chains = chain.get_required_finished_chains()
            # check if the chain requires inputs from other chains
            if required_input_chains:
                undefined_inputs = [x for x in required_input_chains if x not in self.chains]
                if undefined_inputs:
                    error("Chain {} requires an input from the non-existing chain(s): {}".format(chain.get_name(), undefined_inputs))
                if not chain.ready(chain_outputs.get_chain_names()):
                    run_pool.append(chain.get_name())
                    debug("Postponing configuring names for chain {}".format(chain.get_name()))
                    continue
            data_bundle = None
            chain.load_inputs(chain_outputs)
            if required_input_chains:
                # the first component requires an input from another chain -- check if it's loaded
                error("Chain [{}] requires input(s), but none are available.".format(chain.get_name()), chain.inputs is None)
                data_bundle = chain.inputs.get_bundles(chain_names=required_input_chains)
            for c, component in enumerate(chain.components):
                component.load_inputs(data_bundle)
                # configure names
                component.configure_name()
                # configure output dependencies
                if c > 0:
                    chain.components[c-1].add_output_demand(chain.name, component.get_name())
                data_bundle = component.get_outputs()
                chain.outputs = data_bundle

            # set output demand to other chains feeding into this one, to this chain and first component
            if required_input_chains:
                for ch in required_input_chains:
                    self.chains[ch].backtrack_output_demand(chain.get_name(), chain.components[0].get_name(), chain.components[0].consumes)

            chain_outputs.add_bundle(chain.get_outputs(), chain_name=chain.get_name())

    def run(self):
        info("================")
        info("Running pipeline.")
        info("----------------")
        self.sanity_check()
        self.visualize()
        # chain_outputs = {ch: None for ch in self.chains}
        chain_outputs = BundleList()
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

        all_required_inputs = set()
        # check input requirements are satisfiable
        for chain_name, chain in self.chains.items():
            for req_out in chain.get_required_finished_chains():
                if req_out not in self.chains:
                    error("Chain {} requires an output of a non-existent chain: {}".format(chain_name, req_out))
                all_required_inputs.add(req_out)

        # check if endpoints are not dangling
        for chain_name, chain in self.chains.items():
            last_comp = chain.get_components()[-1]
            if not (last_comp.can_be_final or chain_name in all_required_inputs):
                error("Chain [{}] is not piped to another chain and ends in an invalid final component: [{}]".format(
                    chain_name, last_comp.get_full_name()))


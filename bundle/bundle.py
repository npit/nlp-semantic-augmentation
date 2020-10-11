"""Module defining bundle and bundle-list objects
"""
from bundle.datatypes import *
from bundle.datausages import *
from collections import defaultdict
from defs import datatypes
from utils import data_summary, debug, error, info, warning, equal_lengths, as_list

class ResourceIO:
    def __init__(self, dtype, usage, name, chain_name):
        self.dtype = dtype
        self.usage = usage
        self.component_name = name
        self.chain_name = chain_name
    def is_compatible(self, other):
        return other.dtype == self.dtype and other.usage == self.usage


class Produces(ResourceIO):
    def __init__(self, dtype, usage, name, chain_name):
        super().__init__(dtype, usage, name, chain_name)
class Consumes(ResourceIO):
    def __init__(self, dtype, usage, name, chain_name):
        super().__init__(dtype, usage, name, chain_name)

class DataPool:
    """Container class to pass data around
    """

    source_name = None
    chain_name = None

    content_dict = None
    demand = None

    # named links to the next bundle
    linkage = {}
    default_linkage = "overall_output"
    active_linkage = None


    data_per_type = defaultdict(list)
    data_per_usage = defaultdict(list)
    data_per_chain = defaultdict(list)

    requests = defaultdict(list)
    supply = defaultdict(list)

    production = []
    consumption = []
    data = []

    completed_chains = set()

    # chain / component names currently feeding data
    feeder_chains = []
    feeder_components = []

    current_running_chain = None

    reference_data = None

    def add_data_packs(self, datapack_list, source_name):
        for dp in datapack_list:
            dp.chain = self.current_running_chain
            dp.source = source_name
            dp.id = self.make_datapack_id(self.current_running_chain, source_name)
            self.add_data(dp)

    def make_datapack_id(self, chain, source):
        return chain

    def get_outputs(self):
        # get global matches
        res = self.request_data(None, Predictions, "data_pool", usage_matching="subset", reference_data=self.data, must_be_single=False)
        output = {}
        for dp in res:
            output[dp.get_id()] = {}
            output[dp.get_id()]["data"] = dp.data.to_json()
        return output

    def mark_as_reference_data(self):
        """Designate current contents as reference data"""
        self.reference_data = list(range(len(self.data)))
        debug("Data pool marked reference:")
        for x in self.data:
            info(str(x))

    def fallback_to_reference_data(self):
        """Recall to reference"""
        res = []
        for i, dat in enumerate(self.data):
            if i in self.reference_data:
                res.append(dat)
        self.data = res
        debug("Data pool fallback complete -- contents now:")
        for x in self.data:
            info(str(x))

    def add_data(self, data):
        """Add a data pack to the pool"""
        # organize data by type
        data.chain = self.current_running_chain
        self.data_per_type[data.type()].append(data)
        self.data_per_usage[data.usage()].append(data)
        self.data_per_chain[data.chain].append(data)
        self.log_data_supply(data)
        self.data.append(data)

    def log_data_supply(self, data):
        """Log chain data dependencies"""
        self.supply[(data.type(), data.usage(), data.source, data.chain)].append(data)

    def match_usage(self, candidate_usage, usage_requested, usage_matching, usage_exclude=None):
        candidate_usage = as_list(candidate_usage)
        if usage_matching != "all":
            usage_requested = as_list(usage_requested)
        usage_exclude = as_list(usage_exclude)
        if any(x in candidate_usage for x in usage_exclude):
            return False
        if usage_matching == "all":
            return True
        if usage_matching == "exact":
            if not equal_lengths(candidate_usage, usage_requested):
                return False
            return set(candidate_usage) == set(usage_requested)
        elif usage_matching == "subset":
            return all(x in candidate_usage for x in usage_requested)
        elif usage_matching == "any":
            return any(x in usage_requested for x in candidate_usage)
        else:
            error(f"Specified undefined usage matching: {usage_matching}")

    def request_data(self, data_type, usage, client, usage_matching="exact", usage_exclude=None, must_be_single=True,
                     on_error_message="Data request failed:", reference_data=None):
        """Get data from the data pool

        Args:
            data_type (str): Name of datatype
            usage (str): Name or class of usage
            usage_matching (str): How to match the usage, candidates are "exact", "any", "all", "subset". Defaults to "exact"
            client ([type]): [description]
            must_be_single (bool, optional): Singleton enforcer. Defaults to True.
            on_error_message (str, optional): What to print on error. Defaults to "Data request failed:".
            reference_data (list, optional): Data list to draw candidates from. Defaults to None, which is resolved to the current chain feeders.

        Returns:
            [type]: [description]
        """
        # get the data available to the client
        if reference_data is None:
            curr_inputs = self.get_current_inputs()
        else:
            curr_inputs = reference_data
        res = []
        # all to string
        if data_type is not None:
            # data_type = data_type.get_matching_names() if type(data_type) is not str and issubclass(data_type, Datatype) else data_type
            data_type = data_type.name if type(data_type) is not str and issubclass(data_type, Datatype) else data_type
        usage = as_list(usage)
        if any(type(x) is not str and issubclass(x, DataUsage) for x in usage):
            # usage = [x.get_matching_names() if type(x) is not str and issubclass(x, DataUsage) else x for x in usage]
            # flatten
            # usage = [k for x in usage for k in x]
            usage = [x.name if type(x) is not str and issubclass(x, DataUsage) else x for x in usage]
        if usage_exclude is not None:
            usage_exclude = as_list(usage_exclude)
            usage_exclude = [x.name if type(x) is not str and issubclass(x, DataUsage) else x for x in usage_exclude]
        for data in curr_inputs:
            matches_usage = self.match_usage(data.get_usage_names(), usage, usage_matching, usage_exclude)
            if matches_usage and (data_type is None or data.type() in data_type):
                res.append(data)
        if must_be_single:
            if len(res) != 1:
                warning("Examined current inputs for client {client}:")
                for c in curr_inputs:
                    warning(str(c))
                error(on_error_message + f" Requested: {data_type}/{usage}/, matches: {len(res)} candidates but requested a singleton.")
            res = res[0]
        else:
            # else keep all and drop empty ones
            res = drop_empty_datapacks(res)
        return res
            
    def summarize_contents(self):
        for dat in self.data:
            info(dat)
    def get_current_inputs(self):
        """Fetch datapacks currently available from supplying chains / components"""
        res = []
        for dat in self.data:
            if dat.chain in self.feeder_chains:
                if not self.feeder_components or dat.source in self.feeder_components:
                    res.append(dat)
        return res

    def log_data_production(self, productions):
        """Log chain data dependencies"""
        # for prod in productions:
        #     self.production[(data_type, usage)].append((production_name, chain_name))
        self.production.extend(productions)

    def get_input_identifier(self, consumption, source_chains):
        """Figure out the name of the inputs of the requester"""
        matches = []
        incoming_production = [p for p in self.production if p.chain_name in source_chains]
        for c in consumption:
            x = list(filter(lambda x: x.is_compatible(c), incoming_production))
            matches.extend(x)
        aggr_name = "_".join(m.component_name for m in matches)
        return aggr_name

    def log_data_consumption(self, consumptions):
        """Log chain data dependencies"""
        # for con in consumptions:
        #     self.consumption.extend([(con.dtype, con.usage)].append((production_name, chain_name))
        self.consumption.extend(consumptions)

    def on_component_completion(self, chain_name, component_name):
        # remove requirements from the component
        return
        self.remove_fulfilled_demand(chain_name, component_name)

    def on_chain_start(self, chain_name):
        self.current_running_chain = chain_name


    def clear_feeders(self):
        self.feeder_chains.clear()
        self.feeder_components.clear()

    def add_feeders(self, chain_names, component_names):
        """Add feeders"""
        if chain_names is not None:
            for chain_name in chain_names:
                self.feeder_chains.append(chain_name)
        if component_names is not None:
            for component_name in component_names:
                self.feeder_components.append(component_name)

    def on_chain_completion(self, chain_name):
        self.completed_chains.add(chain_name)

    def get_completed_chains(self):
        return self.completed_chains

# -------------------------

    def __str__(self):
        """Generate bundle name

        Returns:
            The generated bundle name
        """
        return "bundle"

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        self.demand = {}

    def remove_fulfilled_demand(self, chain, component):
        """Delete unneeded data
        chain: the chain that just used the bundle
        component: the component that just used the bundle
        """
        debug(f"Demand now is: {self.demand}")
        debug("Removing demand {}-{} from bundle {}".format(chain, component, self.get_full_name()))
        error("Attempted to clear data after run of non-registered chain {}".format(chain), chain not in self.demand)
        error("Attempted to clear data after run of non-registered component {} of chain {}. Registered components are {}.".format(
            component, chain, self.demand[chain]), component not in self.demand[chain])
        self.demand[chain].remove(component)
        if not self.demand[chain]:
            del self.demand[chain]
        if not self.demand:
            debug("Deleting bundle {}.".format(self.get_full_name()))
            if self.vectors:
                del self.vectors
            if self.text is not None:
                del self.text
            if self.labels is not None:
                del self.labels
            if self.indices is not None:
                del self.indices

    def get_full_name(self):
        return "({}|{})".format(self.chain_name, self.source_name)

    def set_demand(self, chain_name, component_name):
        """Update the bundle with a requesting component"""
        # mark the chain / component demand
        if chain_name not in self.demand:
            self.demand[chain_name] = []
        self.demand[chain_name].append(component_name)

    def add_bundle(self, bundle, chain_name=None, linkage_name=None):
        """Add a new bundle to the linked collection. Override chain name, if submitted."""
        if chain_name is not None:
            bundle.set_chain_name(chain_name)
        if linkage_name is None:
            linkage_name = self.get_fallback_linkage()
        tail = self.get_via_condition(lambda x: linkage_name not in x.linkage, enforce_single=True)
        tail.linkage[linkage_name] = bundle
    
    def get_chain_bundles(self, name):
        """Get bundles that belong to the input chain name"""
        res = self.get_via_condition(lambda x: x.get_chain_name() == name)
        error(f"Multi-bundle chain output found: {name}!", len(res) > 1)
        return res[0]

    def summarize_content(self, msg=None, do_propagate=True):
        msg = "" if msg is None else msg + ": "
        msg += f"Bundle: source chain: {self.chain_name}, component {self.source_name}"
        debug(msg)
        if self.vectors is not None:
            data_summary(self.vectors, msg="vectors")
        if self.labels is not None:
            data_summary(self.labels, msg="labels")
        if self.text is not None:
            data_summary(self.text, msg="text")
        if self.indices is not None:
            data_summary(self.indices, msg="text")
        if do_propagate:
            # move along the chain
            if self.has_next():
                self.next().summarize_content(msg)

    # region: getters

    # @staticmethod
    # def append_request_bundle(bundle, client_name, tail_bundle):
    #     """Append a bundle to a client request linked list"""
    #     current_bundle = bundle
    #     while client_name in current_bundle.linkage and current_bundle.linkage[client_name] is not None:
    #         current_bundle = current_bundle.linkage[client_name]
    #     current_bundle.linkage[client_name] = tail_bundle
    #     # add dummy linkage on the tail bundle
    #     Bundle.add_dummy_client_linkage(tail_bundle, client_name)
    #     return bundle

    # endregion

    # region: has-ers
    def has_labels(self):
        return Labels.name in self.data_per_usage

    def has_text(self):
        return Text.name in self.data_per_type

    def has_numerics(self):
        return Numeric.name in self.data_per_type

    def has_indices(self):
        return Indices.name in self.data_per_usage
    # endregion

    # region: setters
    def set_source_name(self, name):
        self.source_name = name

    def set_chain_name(self, name):
        self.chain_name = name
    # endregion

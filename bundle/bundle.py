"""Module defining bundle and bundle-list objects
"""
from bundle.datatypes import Labels, Text, Vectors
from defs import datatypes
from utils import data_summary, debug, error, info, warning


class Bundle:
    """Container class to pass data around
    """
    # data types
    vectors = None
    labels = None
    text = None
    vocabulary = None
    indices = None

    source_name = None
    chain_name = None

    content_dict = None
    demand = None

    # named links to the next bundle
    linkage = {}
    default_linkage = "overall_output"
    active_linkage = None

    def set_active_linkage(self, linkage_name):
        """Set the current linkage"""
        self.active_linkage = linkage_name

    def get_fallback_linkage(self):
        return self.active_linkage

    def get_linkage_bundles(self, linkage_name):
        return self.get_via_condition(lambda x: linkage_name in x.linkage, linkage_name=Bundle.default_linkage)

    def find_linkage_list_head(self, linkage_name):
        all_bundles = self.get_linkage_bundles(linkage_name)
        error(f"No bundles of linkage {linkage_name} found!", not all_bundles)
        candidate_heads = [x for x in all_bundles]
        for b in all_bundles:
            other_bundle = b.linkage[linkage_name]
            if other_bundle is not None:
                candidate_heads.remove(other_bundle)
        error(f"No single head for linkage {linkage_name} found.", len(candidate_heads) != 1)
        return candidate_heads[0]

    @staticmethod
    def print_linkages(bundle, linkage_name=None, visited_bundles=None, print_indent=""):
        if linkage_name is None:
            linkage_name = bundle.get_fallback_linkage()

        all_bundles = bundle.get_bundle_list(linkage_name=Bundle.default_linkage)
        edge_landing_bundles = []
        relevant_bundles = []
        for b in all_bundles:
            if linkage_name in b.linkage:
                relevant_bundles.append(b)
                # track leafs
                if b.linkage[linkage_name] is not None:
                    edge_landing_bundles.append(b.linkage[linkage_name])
                # if not b.has_next(linkage_name) or b.linkage[linkage_name] is None:
                #     candidate_heads.append(b)
        
        if not relevant_bundles:
            info(f"(no linkage bundles for {linkage_name})")
            return
        candidate_heads = [x for x in relevant_bundles if x not in edge_landing_bundles]
        if len(candidate_heads) > 1:
            error(f"Found multiple roots for linkage {linkage_name}: {candidate_heads}")

        candidate_heads[0].print_linkage_sequence(linkage_name)

    def print_linkage_sequence(self, linkage_name=None):
        if linkage_name not in self.linkage:
            warning(f"No linkage: {linkage_name}")
            return
        info(f"Linkage wrt: {linkage_name}:")
        if linkage_name is None:
            linkage_name = self.get_fallback_linkage()
        curr = self
        info(curr)
        while curr.has_next(linkage_name):
            curr = curr.next(linkage_name)
            if curr is None:
                break
            info(curr)
        
    def __len__(self, linkage_name=None):
        ln = 1
        if linkage_name is None:
            linkage_name = self.get_fallback_linkage()
        x = self
        while True:
            if x is None:
                break
            x = x.next(linkage_name)
            ln += 1
        return ln
    
    def has_next(self, active_linkage=None):
        if active_linkage is None:
            active_linkage = self.get_fallback_linkage()
        return active_linkage in self.linkage

    def next(self, active_linkage=None):
        if active_linkage is None:
            active_linkage = self.get_fallback_linkage()
        return self.linkage[active_linkage]

    def clear_data(self, chain_name, component_name):
        self.remove_fulfilled_demand(chain_name, component_name)
        # should not propagate
        # if self.has_next():
        #     self.next().remove_fulfilled_demand(chain_name, component_name)

    def __str__(self):
        """Generate bundle name

        Returns:
            The generated bundle name
        """
        parts = [self.source_name]
        if self.chain_name is not None:
            parts.append(self.chain_name)
        if self.vectors is not None:
            parts.append(Vectors.name)
        if self.labels is not None:
            parts.append(Labels.name)
        if self.text is not None:
            parts.append(Text.name)
        return "_".join(parts)

    def __repr__(self):
        return self.__str__()

    def __init__(self, source_name=None, vectors=None, labels=None, text=None, indices=None):
        self.source_name = source_name
        self.content_dict = {}
        self.demand = {}
        self.active_linkage = self.default_linkage
        self.linkage = {}
        if vectors is not None:
            self.set_vectors(vectors)
        if labels is not None:
            self.set_labels(labels)
        if text is not None:
            self.set_text(text)
        if indices is not None:
            self.set_indices(indices)

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

    def get_bundle_list(x, linkage_name=None):
        """Make list out of the current linked list bundles"""
        if linkage_name is None:
            linkage_name = x.get_fallback_linkage()
        res = [x]
        while x.has_next(linkage_name):
            x = x.next(linkage_name)
            res.append(x)
        return res

    @staticmethod
    def append_request_bundle(bundle, client_name, tail_bundle):
        """Append a bundle to a client request linked list"""
        current_bundle = bundle
        while client_name in current_bundle.linkage and current_bundle.linkage[client_name] is not None:
            current_bundle = current_bundle.linkage[client_name]
        current_bundle.linkage[client_name] = tail_bundle
        # add dummy linkage on the tail bundle
        Bundle.add_dummy_client_linkage(tail_bundle, client_name)
        return bundle

    @staticmethod
    def add_dummy_client_linkage(bundle, client_name):
        """Make a dangling linkage to mark request from a client"""
        if client_name in bundle.linkage:
            error("Attempted to make a dummy linkage on a existing linkage with targets: {self.linkage[client_name]}")
        bundle.linkage[client_name] = None

    def make_request_bundlelist_by_chain_names(self, chain_names, client_name):
        """Get bundles that belong to the input chain names and form them in a request linked list"""
        if type(chain_names) is str:
            chain_names = [chain_names]
        res = None
        for bundle in self.get_bundle_list():
            if bundle.get_chain_name() in chain_names:
                if res is None:
                    res = bundle
                    # add a dummy linkage from the client, with no sibling
                    Bundle.add_dummy_client_linkage(bundle, client_name)
                else:
                    res = Bundle.append_request_bundle(res, client_name, bundle)
        return res

    def get_request_bundlelist(self, client_name):
        """Get bundles that belong to the input chain names and form them in a request linked list"""
        # find head of request list 
        candidate_heads = self.get_via_condition(lambda x: client_name in x.linkage)
        if not candidate_heads:
            error(f"Could not find requested bundles from client {client_name}")
        edge_tails = [x.linkage[client_name] for x in candidate_heads]
        # candidate heads cannot have an incoming link
        candidate_heads = [x for x in candidate_heads if x not in edge_tails]
        if len(candidate_heads) > 1:
            error(f"Found multiple heads for request bundle list from {client_name}: {candidate_heads}")
        return candidate_heads[0]

    def get(self, index):
        """Get the index-th bundle in the bundle list."""
        try:
            return self.get_bundle_list()[index]
        except IndexError:
            error(f"Failed to retrieve the non-existing {index}-th bundle.")

    def get_available(self):
        return list(self.content_dict.keys())

    def get_element(self, element, full_search=False, enforce_single=False, role=None):
        """
        Retrieve a bundle member element based on input restrictions
        """
        res = None
        error(f"Specified both full-search fetching and specified input role(s): {role}", role is not None and full_search)
        if element == datatypes.vectors:
            if full_search:
                res = self.get_via_condition(lambda x: x.vectors is not None)
            else:
                res = self.vectors
        elif element == datatypes.labels:
            if full_search:
                res = self.get_via_condition(lambda x: x.labels is not None)
            else:
                res = self.labels
        elif element == datatypes.text:
            if full_search:
                res = self.get_via_condition(lambda x: x.text is not None)
            else:
                res = self.text
        elif element == datatypes.indices:
            if full_search:
                res = self.get_via_condition(lambda x: x.indices is not None)
            else:
                res = self.indices
        elif element == "source_name":
            if full_search:
                res = [x.source_name for x in self.get_bundle_list()]
            else:
                res = self.source_name
        elif element == "chain_name":
            if full_search:
                res = [x.chain_name for x in self.get_bundle_list()]
            else:
                res = self.chain_name
        else:
            error(f"Undefined element {element} to get from bundle.")

        if res is None:
            return res
        # filter with respect to role
        if role is not None:
            res = self.filter_to_role_instances(res, role, enforce_single)
        if full_search and enforce_single:
            res = self.do_enforce_single(res)
        return res

    def do_enforce_single(self, data):
        if type(data) is list:
            if len(data) != 1:
                data.summarize_content()
                error(f"Enforced single but got {len(data)} data.")
            return data[0]

    @staticmethod
    def filter_to_role_instances(datum, role, enforce_single=False):
        """Limit bundle instances to those of a specified role"""
        res = []
        if datum.has_role(role):
            role_index = datum.roles.index(role)
            res.append(datum.instances[role_index])
        if enforce_single:
            error(f"Enforced single acquisition of role {role} but found {len(res)} matching instances!", len(res) != 1)
            res = res[0]
        return res

    def get_indices(self, full_search=False, enforce_single=False, role=None):
        return self.get_element(datatypes.indices, full_search, enforce_single, role)
    def get_vectors(self, full_search=False, enforce_single=False, role=None):
        return self.get_element(datatypes.vectors, full_search, enforce_single, role)
    def get_text(self, full_search=False, enforce_single=False, role=None):
        return self.get_element(datatypes.text, full_search, enforce_single, role)
    def get_labels(self, full_search=False, enforce_single=False, role=None):
        return self.get_element(datatypes.labels, full_search, enforce_single, role)
    def get_source_name(self, full_search=False, enforce_single=False, role=None):
        return self.get_element("source_name", full_search, enforce_single, role)
    def get_chain_name(self, full_search=False, enforce_single=False, role=None):
        return self.get_element("chain_name", full_search, enforce_single, role)

    def get_via_condition(self, cond, linkage_name=None, enforce_single=False):
        """Return bundles matching an input condition"""
        if linkage_name is None:
            linkage_name = self.get_fallback_linkage()
        res = []
        # start from the base node
        x = self
        while True:
            if cond(x):
                res.append(x)
            if not x.has_next(linkage_name):
                break
            x = x.next(linkage_name)
            if x is None:
                break
        if enforce_single:
            error(f"Enforced single acquisition of condition {cond} but found {len(res)} matching instances!", len(res) != 1)
            res = res[0]
        return res
    # endregion

    # region: has-ers
    def has_labels(self):
        return self.has_condition(lambda x: x.labels is not None)

    def has_source_name(self):
        return self.has_condition(lambda x: x.source_name is not None)

    def has_vectors(self):
        return self.has_condition(lambda x: x.vectors is not None)

    def has_text(self):
        return self.has_condition(lambda x: x.text is not None)

    def has_condition(self, cond):
        x = self.get_via_condition(cond)
        return x is not None and len(x) > 0

    def has_indices(self):
        return self.has_condition(lambda x: x.indices is not None)
    # endregion

    # region: setters
    def set_text(self, text):
        self.text = text
        self.content_dict["text"] = text

    def set_vectors(self, vectors):
        self.content_dict["vectors"] = vectors
        self.vectors = vectors

    def set_labels(self, labels):
        self.labels = labels
        self.content_dict["labels"] = labels

    def set_indices(self, indices):
        self.indices = indices
        self.content_dict["indices"] = indices

    def set_source_name(self, name):
        self.source_name = name

    def set_chain_name(self, name):
        self.chain_name = name

    def set_train(self, idx):
        self.train_idx = idx

    def set_test(self, idx):
        self.test_idx = idx
    # endregion

"""Module defining bundle and bundle-list objects
"""
from bundle.datatypes import Labels, Text, Vectors
from defs import datatypes
from utils import data_summary, debug, error, info


class Bundle:
    """Data container class to pass data around
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

    def __init__(self, source_name=None, vectors=None, labels=None, text=None, indices=None):
        self.source_name = source_name
        self.content_dict = {}
        self.demand = {}
        if vectors is not None:
            self.content_dict["vectors"] = vectors
            self.vectors = vectors
        if labels is not None:
            self.content_dict["labels"] = labels
            self.labels = labels
        if text is not None:
            self.content_dict["text"] = text
            self.text = text
        if indices is not None:
            self.content_dict["indices"] = indices
            self.indices = indices

    def clear_data(self, chain_name, component_name):
        """Delete unneeded data
        chain_name: the chain that just used the bundle
        component_name: the component that just used the bundle
        """
        debug("Removing demand {}-{} from bundle {}".format(chain_name, component_name, self.get_full_name()))
        error("Attempted to clear data after run of non-registered chain {}".format(chain_name), chain_name not in self.demand)
        error("Attempted to clear data after run of non-registered component {} of chain {}. Registered components are {}.".format(
            component_name, chain_name, self.demand[chain_name]), component_name not in self.demand[chain_name])
        self.demand[chain_name].remove(component_name)
        if not self.demand[chain_name]:
            del self.demand[chain_name]
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
        if chain_name not in self.demand:
            self.demand[chain_name] = []
        self.demand[chain_name].append(component_name)

    def summarize_content(self, msg=None):
        msg = "" if msg is None else msg + ": "
        msg += "Bundle source chain: {}, component {}:".format(self.chain_name, self.source_name)
        debug(msg)
        if self.vectors is not None:
            data_summary(self.vectors, msg="vectors")
        if self.labels is not None:
            data_summary(self.labels, msg="labels")
        if self.text is not None:
            data_summary(self.text, msg="text")
        if self.indices is not None:
            data_summary(self.indices, msg="text")

    # region: getters

    def get_available(self):
        return list(self.content_dict.keys())

    def get_element(self, element, role=None):
        """Retrieve element based on input type and role"""
        res = None
        if element == datatypes.vectors:
            res = self.vectors
        elif element == datatypes.labels:
            res = self.labels
        elif element == datatypes.text:
            res = self.text
        elif element == datatypes.indices:
            res = self.indices
        else:
            error(f"Undefined element {element} to get from bundle.")
        # filter with respect to role
        if res is not None and role is not None:
            # make sure indices are set
            if self.indices is None:
                error(f"Requested role {role} but current bundle {self.get_full_name()} has no indices!")
            try:
                role_index = self.indices.roles.index(role)
                res = res.instances[role_index]
            except ValueError:
                error(f"Role {role} not in bundle indices roles: {self.indices.roles}")
            except IndexError:
                error(f"Index of requested role {role} is {role_index} but bundle {element} elements contain {len(res.instances)} instances.")
        return res

    def get_vectors(self, role=None):
        return self.get_element(datatypes.vectors, role)

    def get_indices(self, role=None):
        return self.get_element(datatypes.indices, role)

    def get_text(self, role=None):
        return self.get_element(datatypes.text, role)

    def get_labels(self, role=None):
        return self.get_element(datatypes.labels, role)

    def get_source_name(self):
        return self.source_name

    def get_chain_name(self):
        return self.chain_name
    # endregion

    # region: has-ers
    def has_labels(self):
        return self.labels is not None

    def has_source_name(self):
        return self.source_name is not None

    def has_vectors(self):
        return self.vectors is not None

    def has_text(self):
        return self.text is not None

    def has_indices(self):
        return self.text is not None
    # endregion

    # region: setters
    def set_text(self, text):
        self.text = text

    def set_vectors(self, vectors):
        self.vectors = vectors

    def set_labels(self, labels):
        self.labels = labels

    def set_indices(self, indices):
        self.indices = indices

    def set_source_name(self, name):
        self.source_name = name

    def set_chain_name(self, name):
        self.chain_name = name

    def set_train(self, idx):
        self.train_idx = idx

    def set_test(self, idx):
        self.test_idx = idx
    # endregion


class BundleList:
    """Class to represent a collection of bundles
    """
    bundles = None

    def __len__(self):
        return len(self.bundles) if self.bundles else 0

    def __init__(self, input_list=None):
        self.bundles = [] if input_list is None else input_list

    def clear_data(self, chain_name, component_name):
        for bundle in self.bundles:
            bundle.clear_data(chain_name, component_name)

    def summarize_content(self, msg=""):
        """Summarize all bundlelist bundles"""
        if not self.bundles:
            info("Empty bundle list.")
            return
        if msg:
            msg += ": "
        msg += "Summary of {}-long bundle list".format(len(self.bundles))
        info(msg)
        for bundle in self.bundles:
            bundle.summarize_content()

    def set_demand(self, chain_name, component_name):
        for bundle in self.bundles:
            bundle.set_demand(chain_name, component_name)

    def add_bundle(self, bundle, chain_name=None):
        """Add a bundle to the collection. Override name, if submitted."""
        if type(bundle) is BundleList:
            error("Attempted to add a bundle list to a bundle list.")
        if chain_name is not None:
            bundle.set_chain_name(chain_name)
        self.bundles.append(bundle)

    # region: getters
    def get(self, index):
        """Get the index-th bundle in the bundle list."""
        return self.bundles[index]

    def get_element(self, element, only_single=False, role=None):
        res = [x.get_element(element, role) for x in self.bundles]
        # non-None
        res_idxs = [i for i in range(len(res)) if res[i] is not None]
        res = [r for r in res if r is not None]
        if only_single:
            # ensure single
            names = list(map(lambda x: x.get_source_name(), [self.bundles[i] for i in res_idxs]))
            if len(res) > 1:
                error(f"Requested single-bundle {element} {role} data but multiple exist in bundles {names}")
            res = res[0]
        return res

    def get_source_name(self):
        return [x.get_source_name() for x in self.bundles]

    def get_vectors(self, single=False, role=None):
        return self.get_element(datatypes.vectors, single, role)

    def get_indices(self, single=False, role=None):
        return self.get_element(datatypes.indices, single, role)

    def get_labels(self, single=False, role=None):
        return self.get_element(datatypes.labels, single, role)

    def get_texts(self, single=False, role=None):
        return self.get_element(datatypes.texts, single, role)

    def get_names(self):
        return [x.get_source_name() for x in self.bundles]

    def get_chain_names(self):
        return [x.get_chain_name() for x in self.bundles]

    def get_bundles(self, chain_names):
        """Get bundles that belong to the input chain names"""
        if len(chain_names) == 1:
            return self.get_bundle(chain_names[0])
        res = BundleList()
        for bundle in self.bundles:
            if bundle.get_chain_name() in chain_names:
                res.add_bundle(bundle)
        return res

    def get_bundle(self, name):
        """Get bundles that belong to the input chain name"""
        for bundle in self.bundles:
            if bundle.get_chain_name() == name:
                return bundle
        return None

    def get_bundle_like(self, element, role=None, single=False):
        """Retrieve a bundle based on specified characteristics"""
        res = []
        for bundle in self.bundles:
            if bundle.get_element(element, role=role) is not None:
                res.append(bundle)
        if single:
            error(f"Requested a single bundles with {element} and role: {role} but {len(res)} were found!", len(res)>1)
            res = res[0]
        return res
    # endregion

    # region : #has'ers - enforce uniqueness
    def has_indices(self):
        return any([x.get_indices() is not None for x in self.bundles])
    def has_labels(self):
        return any([x.get_labels() is not None for x in self.bundles])
    def has_chain_name(self):
        return any([x.get_chain_name() is not None for x in self.bundles])
    def has_source_name(self):
        return any([x.get_source_name() is not None for x in self.bundles])
    def has_vectors(self):
        return any([x.get_vectors() is not None for x in self.bundles])
    def has_text(self):
        return any([x.get_text() is not None for x in self.bundles])
    # endregion

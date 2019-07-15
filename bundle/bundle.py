from bundle.datatypes import Text, Labels, Vectors
from utils import info, data_summary, error, debug

"""Data container class to pass data around
"""
class Bundle:
    # data types
    vectors = None
    labels = None
    text = None
    vocabulary = None

    source_name = None
    chain_name = None

    content_dict = None
    demand = None

    def __str__(self):
        parts = [self.source_name]
        if self.chain_name is not None:
            parts.append(self.chain_name)
        if self.vectors is not None:
            parts.append(Vectors.name)
        if self.vectors is not None:
            parts.append(Labels.name)
        if self.vectors is not None:
            parts.append(Text.name)
        return "_".join(parts)

    def __init__(self, source_name=None, vectors=None, labels=None, text=None):
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

    def clear_data(self, chain_name, component_name):
        """Delete unneeded data
        chain_name: the chain that just used the bundle
        component_name: the component that just used the bundle
        """
        debug("Removing demand {}-{} from bundle {}".format(chain_name, component_name, self.get_full_name()))
        error("Attempted to clear data after run of non-registered chain {}".format(chain_name), chain_name not in self.demand)
        error("Attempted to clear data after run of non-registered component {} of chain {}".format(
            component_name, chain_name), component_name not in self.demand[chain_name])
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
            pass

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

    # region: getters

    def get_available(self):
        return list(self.content_dict.keys())

    def get_labels(self):
        return self.labels

    def get_source_name(self):
        return self.source_name

    def get_chain_name(self):
        return self.chain_name

    def get_vectors(self):
        return self.vectors

    def get_text(self):
        return self.text
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
    # endregion

    # region: setters
    def set_text(self, text):
        self.text = text
    def set_vectors(self, vectors):
        self.vectors = vectors
    def set_labels(self, labels):
        self.labels = labels
    def set_source_name(self, name):
        self.source_name = name
    def set_chain_name(self, name):
        self.chain_name = name
    # endregion


"""Class to represent a collection of bundles
"""
class BundleList:
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
    def single_filter(self, data, do_filter):
        if do_filter:
            data = [x for x in data if x is not None]
            error("Requested single-bundle data but multiple exist in bundles {}".format(
                map(lambda x: x.get_source_name(), data)), len(data) > 1)
            data = data[0]
        return data

    def get_source_name(self):
        return [x.get_source_name() for x in self.bundles]

    def get_vectors(self, single=False):
        res = [x.get_vectors() for x in self.bundles]
        return self.single_filter(res, single)

    def get(self, index):
        return self.bundles[index]

    def get_labels(self, single=False):
        res = [x.get_labels() for x in self.bundles]
        return self.single_filter(res, single)

    def get_texts(self, single=False):
        res = [x.get_text() for x in self.bundles]
        return self.single_filter(res, single)

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
    # endregion

    # region : #has'ers - enforce uniqueness
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

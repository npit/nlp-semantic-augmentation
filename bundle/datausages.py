from bundle.datatypes import *
import numpy as np
from utils import as_list, warning, align_index

class DataUsage:
    """Abstract class for types of data utilization"""
    name = None
    valid_datatypes = None
    def __init__(self, valid_datatypes):
        self.valid_datatypes = valid_datatypes

    @classmethod
    def get_subclasses(cls):
        """Get a list of names matching the datatype"""
        ret =  [cls.name]
        for scls in cls.__subclasses__():
            ret.extend(scls.get_subclasses())
        return ret

class Indices(DataUsage):
    """Indexes to a collection of data with train/test roles"""
    name = "indices"

    # how many objects each "instance" is composed of
    elements_per_instance = None
    # train/test roles
    instances = None
    tags = None

    def add_tags(self, tags):
        """Add tag metadata"""
        self.tags = tags
        if len(self.tags) != len(self.instances):
            warning(f"Added {len(self.tags)} tags on indexes composed of {len(self.instances)} instances.")

    def to_json(self):
        res = {"instances":[], "tags":[]}
        for inst in self.instances:
            res["instances"].append(inst.tolist())
        return res

    def apply_mask(self, surviving):
        """Apply a boolean deletion mask and realign all indexes
        """
        output = []
        for inst in self.instances:
            binary_mask = [i in surviving for i in inst]
            ind  = align_index(inst, binary_mask, mask_shows_deletion=False)
            output.append(ind)
        return self.__class__(output, self.tags)


    def __init__(self, instances, tags, epi=None):
        # only numbers
        super().__init__([Numeric])
        instances = as_list(instances)
        tags = as_list(tags)
        if epi is None:
            epi = [np.ones((len(ind),), np.int32) for ind in instances]
        self.elements_per_instance = epi 

        self.instances = []
        self.tags = []
        for i in range(len(instances)):
            inst = instances[i]
            if len(inst) == 0:
                continue
            self.instances.append(np.asarray(inst))
            try:
                tag = tags[i]
                self.tags.append(tag)
            except (TypeError, IndexError):
                pass

    def get_train_test(self):
        """Get train/test indexes"""
        train, test = np.ndarray((0,), np.int32), np.ndarray((0,), np.int32)
        for i in range(len(self.instances)):
            if self.tags[i] == defs.roles.train:
                train = np.append(train, self.instances[i])
            elif self.tags[i] == defs.roles.test:
                test = np.append(test, self.instances[i])
        return train, test

    def summarize_content(self):
        """Print a summary of the data"""
        return f"{self.name} {len(self.instances)} instances, {self.tags} tags"

    def has_role(self, role):
        """Checks for the existence of a role in the avail. instances"""
        if not self.tags:
            return False
        return role in self.tags

    def get_train_instances(self):
        return self.get_tag_instances(defs.roles.train)
    def get_test_instances(self):
        return self.get_tag_instances(defs.roles.test)

    def get_overlapping(self, input_idx, input_tag):
        """Fetch indexes and usages overlapping with the input indexes
        """
        out_inst, out_tag = [], []
        for i in range(len(self.instances)):
            inst, tag = self.instances[i], self.tags[i]
            # skip identical tag
            if tag == input_tag:
                continue
            # get overlap
            overlap = np.intersect1d(input_idx, inst)
            if len(overlap) > 0:
                out_inst.append(overlap)
                out_tag.append(tag)
        return out_inst, out_tag

    def get_tag_instances(self, role, must_be_single=True, must_exist=True):
        """Retrieve instances associated with the input role"""
        if not self.has_role(role):
            error(f"Required tag {role} not found in indices! Available: {self.tags}", must_exist)
            return np.ndarray((0,), dtype=np.int32)
        role_idx = [i for i in range(len(self.tags)) if self.tags[i] == role]
        inst = [self.instances[i] for i in role_idx]
        if must_be_single:
            error(f"Found {len(inst)} but require a single set of instances with role {role}", len(inst) != 1)
            inst = inst[0]
        return inst

    def equals(self, other):
        if not len(self.instances) == len(other.instances):
            return False
        for i in range(len(self.instances)):
            if not np.array_equal(self.instances[i], other.instances[i]):
                return False
        return True


    def add_instance(self, idx, tag):
        self.instances.append(idx)
        self.tags.append(tag)
        error(f"Adding role {tag} to index with non-equal instances: {len(self.instances)} and tags {len(self.tags)}", len(self.instances) != len(self.tags))

    def __str__(self):
        dat = "|".join([f"{x.shape} : {t}" for (x, t) in zip(self.instances, self.tags)])
        return f"{self.name} : [{dat}]" 
    def __repr__(self):
        return self.__str__()

class Predictions(Indices):
    """Usage for denoting learner predictions"""
    name = "predictions"
    labelset = None

    def add_ground_truth(self, labelset):
        self.labelset = labelset

    def to_json(self):
        res = super().to_json()
        res["tags"] = []
        if self.tags is not None:
            for tag in self.tags:
                res["tags"].append(tag)
        res["labelset"] = []
        if self.labelset is not None:
            res["labelset"] = list(sorted(self.labelset))
        return res

class GroundTruth(DataUsage):
    """Class for abstract ground truth objects"""
    # whether counting / enumerating the GT 
    name = "groundtruth"
    discrete = None
    def __init__(self, discrete=False):
        self.discrete = discrete

    # @classmethod
    # def get_matching_names(cls):
    #     """Get a list of names matching the datatype"""
    #     return [GroundTruth.name ]

class Labels(GroundTruth):
    """Class for enumerable discrete ground truth"""
    name = "labels"
    labelset = None
    multilabel = None

    def __init__(self, labelset=None, multilabel=None):
        super().__init__(discrete=True)
        self.labelset = labelset
        self.multilabel = multilabel

    def is_multilabel(self):
        return self.multilabel

    def to_json(self):
        res = super().to_json()
        res["labelset"] = self.labelset
        res["multilabel"] = int(self.multilabel)
        return res

class DataPack:
    """Objects with data associated with data usages"""
    data = None
    usages = None
    chain = "NO_CHAIN"
    source = "NO_SOURCE"
    id = "NO_ID"

    def to_json(self):
        res = {"usages": []}
        res["data"] = self.data.to_json()
        for us in self.usages:
            res["usages"].append(us.to_json())
        return res


    def __init__(self, data, usage=None, source=None, chain=None):
        self.data = data
        self.usages = []
        if usage is not None:
            usage = as_list(usage)
            self.usages.extend(usage)
        if source is not None:
            self.source = source
        if chain is not None:
            self.chain = chain

    def add_usage(self, us):
        error("Attempted to add duplicate usage {us.name} in datapack!", us in self.usages)
        self.usages.append(us)

    def usage(self):
        return "_".join([x.name for x in self.usages])

    def get_id(self):
        return self.data.name + "|" + self.usage()

    def get_datatype(self):
        """Get type of data"""
        return self.data.name

    def get_usage_names(self):
        return [x.name for x in self.usages]

    def get_usage(self, desired_usage, allow_multiple=False, allow_superclasses=True):
        res = []
        matches = lambda x, desired_usage: x == desired_usage or (allow_superclasses and issubclass(x, desired_usage))
        for u in self.usages:
            if matches(type(u), desired_usage):
                res.append(u)
        if res:
            if not allow_multiple:
                error(f"Requested single usage of type {desired_usage} but {len(res)} were found", len(res) > 1)
                res = res[0]
        return res


    @staticmethod
    def make(data, usage_class):
        cl = get_data_class(data)
        dat = cl(data)
        usage = usage_class()
        dp = DataPack(data=dat, usage=usage)
        return dp

    def __str__(self):
        return f"{self.get_datatype()}|{self.usage()}|{self.source}|{self.chain}"
    def __repr__(self):
        return str(self)

def drop_empty_datapacks(dps):
    res = []
    for dp in dps:
        # exempt dummy data
        if type(dp.data) is DummyData:
            res.append(dp)
        else:
            retain = False
            inst =  dp.data.instances
            for x in inst:
                if len(x) >0:
                    retain = True
            if retain:
                res.append(dp)
    return res
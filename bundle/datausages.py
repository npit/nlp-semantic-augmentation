from bundle.datatypes import *
import numpy as np
from utils import as_list

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
    roles = None
    instances = None

    def __init__(self, instances, epi=None, roles=None):
        # only numbers
        super().__init__([Numeric])
        if epi is None:
            epi = [np.ones((len(ind),), np.int32) for ind in instances]
        self.elements_per_instance = epi 

        self.instances = []
        self.roles = []
        instances = as_list(instances)
        if roles is None:
            roles = []
        roles = as_list(roles)
        for i in range(len(instances)):
            try:
                role = roles[i]
            except IndexError:
                role = None

            inst = instances[i]
            if len(inst) == 0:
                continue
            self.instances.append(inst)
            if role is not None:
                self.roles.append(role)

    def get_train_test(self):
        """Get train/test indexes"""
        train, test = np.ndarray((0,), np.int32), np.ndarray((0,), np.int32)
        for i in range(len(self.instances)):
            if self.roles[i] == defs.roles.train:
                train = np.append(train, self.instances[i])
            elif self.roles[i] == defs.roles.test:
                test = np.append(test, self.instances[i])
        return train, test


    def summarize_content(self):
        """Print a summary of the data"""
        return f"{self.name} {len(self.instances)} instances, {self.roles} roles"

    def has_role(self, role):
        """Checks for the existence of a role in the avail. instances"""
        if not self.roles:
            return False
        return role in self.roles

    def get_train_instances(self):
        return self.get_role_instances(defs.roles.train)
    def get_test_instances(self):
        return self.get_role_instances(defs.roles.test)

    def get_role_instances(self, role, must_be_single=True, must_exist=True):
        """Retrieve instances associated with the input role"""
        if not self.has_role(role):
            error(f"Required role {role} not found in indices!", must_exist)
            return np.ndarray((0,), dtype=np.int32)
        role_idx = [i for i in range(len(self.roles)) if self.roles[i] == role]
        inst = [self.instances[i] for i in role_idx]
        if must_be_single:
            error(f"Found {len(inst)} but require a single set of instances with role {role}", len(inst) != 1)
            inst = inst[0]
        return inst

    def append_index(self, idx, role=None):
        """Add another set of indices"""
        self.instances.append(idx)
        self.roles.append(role)


class Predictions(Indices):
    """Usage for denoting learner predictions"""
    name = "predictions"
    pass

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


class DataPack:
    """Objects with data associated with data usages"""
    data = None
    usages = None
    chain = "NO_CHAIN"
    source = "NO_SOURCE"
    id = "NO_ID"

    def __init__(self, data, usage=None, source=None, chain=None):
        self.data = data
        self.usages = []
        if usage is not None:
            self.usages.append(usage)
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

    def type(self):
        """Get type of data"""
        return self.data.name

    def get_usage_names(self):
        return [x.name for x in self.usages]

    def get_usage(self, usage_name):
        if type(usage_name) is not str and issubclass(usage_name, DataUsage):
            usage_name = usage_name.name

        for u in self.usages:
            if u.name == usage_name:
                return u
        return None


    @staticmethod
    def make(data, usage_class):
        cl = get_data_class(data)
        dat = cl(data)
        usage = usage_class()
        dp = DataPack(data=dat, usage=usage)
        return dp

    def __str__(self):
        return f"{self.type()}|{self.usage()}|{self.source}|{self.chain}"
    def __repr__(self):
        return str(self)

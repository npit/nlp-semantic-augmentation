"""Module defining bundle datatypes"""

import numpy as np
from numbers import Number
from collections import OrderedDict

import defs
from defs import avail_roles, roles_compatible
from utils import error, data_summary, is_collection


class Datatype:
    """Abstract class for a data type"""
    # whether the data should be used for training or testing
    roles = None
    # whether the data should be treaded as data or ground truth

    # variable for the distinct data units
    instances = None

    def __init__(self, instances):
        self.instances = instances

    def get_instance(self, instance_idx):
        try:
            if type(self.instances) is np.ndarray:
                return self.instances[instance_idx]
            else:
                return [self.instances[i] for i in instance_idx]
        except KeyError:
            return None

    # def get_instance_index(self, idx):
    #     llen = len(self.instances)
    #     if llen <= idx:
    #         error(f"Requested index {idx} from {self.name} datatype which has only (llen)")
    #     return self.instances[idx]

    # def has_role(self, role):
    #     """Checks for the existence of a role in the avail. instances"""
    #     if not self.roles:
    #         return False
    #     return role in self.roles

    # def get_train_role_indexes(self):
    #     """Retrieve instance indexes with a training role"""
    #     return self.get_role_indexes(defs.roles.train)

    # def get_test_role_indexes(self):
    #     """Retrieve instance indexes with a testing role"""
    #     return self.get_role_indexes(defs.roles.test)
        
    # def get_role_indexes(self, inp_role):
    #     res = []
    #     for r, role in enumerate(self.roles):
    #         if role == inp_role:
    #             res.append(r)
    #     return res

    # def summarize_content(self):
    #     data_summary(self, self.name)

    def append_instance(self, inst):
        """Append another instance object"""
        self.instances = np.append(self.instances, inst, axis=0)


class Text(Datatype):
    """Textual data"""
    name = "text"
    vocabulary = None

    def __init__(self, inst, vocab=None):
        super().__init__(inst)
        self.vocabulary = vocab

    @staticmethod
    def get_strings(data):
        """Get text from the dataset outputs"""
        return [" ".join(item["words"]) for item in data]

class Numeric(Datatype):
    name = "numeric"
    def __init__(self, inst):
        super().__init__(inst)


def get_data_class(inputs):
    # text info dict
    if type(inputs) in [dict, OrderedDict] and "words" in inputs:
        return Text
    if is_collection(inputs[0]):
        return get_data_class(inputs[0])
    if type(inputs[0]) in (str, np.str_):
        return Text
    elif isinstance(inputs[0], Number) or type(inputs[0]) is np.ndarray:
        return Numeric
    else:
        error(f"Unsupported type of data: {inputs[0]}")

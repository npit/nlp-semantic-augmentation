"""Module defining bundle datatypes"""

import numpy as np
from numbers import Number
from collections import OrderedDict

import defs
from utils import error, data_summary, is_collection


class Datatype:
    """Abstract class for a data type"""
    # variable for the distinct data units
    instances = None
    name = None
    def __init__(self, instances):
        self.instances = instances

    def __str__(self):
        return self.name + f" {len(self.instances)}"

    def get_slice(self, instance_idx):
        try:
            if type(self.instances) is np.ndarray:
                return self.instances[instance_idx]
            else:
                return [self.instances[i] for i in instance_idx]
        except KeyError:
            return None

    def to_json(self):
        res = {}
        for i, inst in enumerate(self.instances):
            if type(inst) is np.ndarray:
                inst = inst.tolist()
            res[i] = inst
        return res

    def append_instance(self, inst):
        """Append another instance object"""
        self.instances = np.append(self.instances, inst, axis=0)
    
    @classmethod
    def get_subclasses(cls):
        """Get a list of names matching the datatype"""
        ret =  [cls.name]
        for scls in cls.__subclasses__():
            ret.extend(scls.get_subclasses())
        return ret


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
    @staticmethod
    def get_words(data):
        """Get text from the dataset outputs"""
        return [item["words"] for item in data]

class Numeric(Datatype):
    name = "numeric"
    def __init__(self, inst):
        super().__init__(inst)

class Dictionary(Datatype):
    name = "dict"
    def __init__(self, inst):
        super().__init__(inst)

    def to_json(self):
        return self.instances

class DummyData(Datatype):
    name = "dummy"
    def __init__(self):
        super().__init__([])


def get_data_class(inputs):
    """
    Get datatype class of the input
    """
    # input is already a datatype
    if issubclass(type(inputs), Datatype):
        return type(inputs)
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

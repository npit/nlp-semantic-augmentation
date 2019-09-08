from collections import OrderedDict
from copy import deepcopy

from utils import error, info


class VariableConf(OrderedDict):
    id = None

    @staticmethod
    def get_copy(instance):
        new_instance = deepcopy(OrderedDict(instance))
        new_instance = VariableConf(new_instance)
        new_instance.id = instance.id
        new_instance.ddict = deepcopy(instance.ddict)
        return new_instance

    def __init__(self, existing=None):
        if existing is not None:
            error("Ordered dict required for variable config. ",
                  type(existing) != OrderedDict)
            super().__init__(existing)
        else:
            super().__init__()
        self.id = ""
        self.ddict = {}

    def add_variable(self, keys, value):
        info("Setting variable field: {} / value: {} -- current conf id: {}".
             format(keys, value, self.id))
        conf = self
        for k, key in enumerate(keys[:-1]):
            if key not in conf:
                error(
                    "Key not present in configuration and it's not a parent component name.",
                    k != len(keys) - 2)
                conf[key] = {}
            conf = conf[key]
        if keys[-1] in conf:
            error("Variable key already in configuration!")
        conf[keys[-1]] = value

        # use the last key for the id -- revisit if there's ambiguity
        # self.id = "_".join(keys) + "_" + str(value)
        if type(value) == list:
            strvalue = "_".join(map(str, value))
        else:
            strvalue = str(value)
        strvalue = strvalue.replace("/", "_")

        if len(self.id) > 0:
            self.id += "_"
        self.id += keys[-1] + "_" + strvalue
        info("Final conf id: {}".format(self.id))
        self.ddict[keys[-1]] = value

    def __str__(self):
        return self.id + " : " + super().__str__()

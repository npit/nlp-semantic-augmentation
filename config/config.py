"""Module for abstract configuration
"""
from copy import deepcopy

import defs
from utils import error


class Configuration:
    """Class representing an abstract configuration of a single runnable component

    Includes global-level configs as well.
    """
    component = None
    explicit_model_path = None

    allow_output_deserialization = None
    allow_model_deserialization = None

    def __init__(self, config):
        self.conf = config
        self.config_objects = []
        self.config_object_names = []
        # component-level deserialization flags
        if config is not None:
            self.allow_output_deserialization = self.get_value("allow_output_deserialization", base=config, default=None)
            self.allow_model_deserialization = self.get_value("allow_model_deserialization", base=config, default=None)

        if config is not None:
            self.explicit_model_path = self.get_value("model_path", default=None, base=config)

    def output_deserialization_allowed(self):
        return self.allow_output_deserialization == True or (self.allow_output_deserialization is None and self.misc.allow_output_deserialization)
    def model_deserialization_allowed(self):
        return self.allow_model_deserialization == True or (self.allow_model_deserialization is None and self.misc.allow_model_deserialization)

    def add_config_object(self, conf_name, conf_object):
        """Add a configuration sub-object to the current configuration object
        """
        new_object = deepcopy(conf_object)
        self.__setattr__(conf_name, new_object)
        self.config_objects.append(new_object)
        self.config_object_names.append(conf_name)

    def merge_other_config(self, other):
        for conf_name, conf_object in zip(other.config_object_names, other.config_objects):
            self.add_config_object(conf_name, conf_object)

    def get_value(self, name, default=None, base=None, expected_type=None):
        if base is None:
            base = self.conf
        value = base[name] if name in base and base[name] is not None else default
        if expected_type is not None and value is not None:
            if type(value) is not expected_type:
                error("Argument {} got value {} which is of type {}, but {} is required."
                      .format(name, value, type(value), expected_type))
        return value

    def has_value(self, name, base=None):
        if base is None:
            base = self.conf
        return name in base and base[name] is not None and base[name] != defs.alias.none

    # all available components
    def get_copy(self):
        """Get a copy of the config object

        This omits the logger object, which causes problems to the copying procedure.
        """
        c = Configuration(self.conf)
        c.merge_other_config(self)
        return c

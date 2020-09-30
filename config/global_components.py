"""Module for global-level configuration
"""
import random
from os.path import join

import nltk

from config.config import Configuration
from utils import datetime_str, warning


class print_conf(Configuration):
    conf_key_name = "print"
    folds = None

    def __init__(self, config=None):
        """Constructor for the printing component configuration"""
        super().__init__(config)
        if config is None:
            return
        self.folds = self.get_value("folds", base=config, default=False)
        self.training_progress = self.get_value("training_progress", base=config, default=True)
        self.log_level = self.get_value("log_level", base=config, default="info")


class misc_conf(Configuration):
    conf_key_name = "misc"
    seed = None
    keys = {}
    csv_separator = ","
    independent_component = None
    allow_deserialization = None
    allow_model_loading = None

    def __init__(self, config=None):
        """Constructor for the miscellaneous configuration"""
        if config is None:
            return
        super().__init__(config)
        if self.has_value("keys", base=config):
            for kname, kvalue in config['keys'].items():
                self.keys[kname] = kvalue
        self.independent = self.get_value("independent_component", base=config, default=False)
        self.allow_deserialization = self.get_value("deserialization_allowed", base=config, default=True)
        self.allow_prediction_loading = self.get_value("prediction_loading_allowed", base=config, default=False)
        self.allow_model_loading = self.get_value("model_loading_allowed", base=config, default=False)
        self.csv_separator = self.get_value("csv_separator", base=config, default=",")
        self.run_id = self.get_value("run_id", base=config, default="run_" + datetime_str())

        if not self.has_value("keys", base=config):
            self.seed = random.randint(0, 5000)
            warning("No random seed submitted, randomly generated: {}".format(self.seed))
        else:
            self.seed = self.get_value("seed", base=config)


class folders_conf(Configuration):
    conf_key_name = "folders"
    run = None
    results = None
    serialization = None
    raw_data = None
    logs = None

    def __init__(self, config=None):
        """Constructor for the folders configuration"""
        if config is None:
            return
        super().__init__(config)
        self.run = config["run"]
        self.results = join(self.run, "results")
        self.serialization = self.get_value("serialization", base=config, default="serialization")
        self.raw_data = self.get_value("raw_data", base=config, default="raw_data")
        nltk_data_path = self.get_value("nltk", base=config, default=join(self.raw_data, "nltk"))
        # set nltk data folder
        nltk.data.path = [nltk_data_path]

class deploy_conf(Configuration):
    conf_key_name = "deploy"
    def __init__(self, config=None):
        """Constructor for the deployment configuration"""
        if config is None:
            return
        super().__init__(config)


global_component_classes = [print_conf, misc_conf, folders_conf, deploy_conf]

import os
from copy import deepcopy
from os.path import join, exists
import logging
import random
import yaml
import utils
import defs
import nltk
from utils import need, error, info, ordered_load, warning
import shutil

from component.chain import Chain
from component.pipeline import Pipeline

from collections import OrderedDict



class Config:
    seed_file = "seed.txt"
    # logger_name = "root"
    logger = None
    conf = {}
    run_id = None

    chains = []

    def get_copy(self):
        """Get a copy of the config object

        This omits the logger object, which causes problems to the copying procedure."""
        c = Config()
        c.dataset = deepcopy(self.dataset)
        c.representation = deepcopy(self.representation)
        c.transform = deepcopy(self.transform)
        c.semantic = deepcopy(self.semantic)
        c.learner = deepcopy(self.learner)
        c.train = self.train
        c.misc = deepcopy(self.misc)
        c.print = deepcopy(self.print)
        c.folders = deepcopy(self.folders)

        return c

    class manip:
        name = None
        times = None

    class dataset_conf:
        name = None
        prepro = None
        data_limit = [None, None]
        class_limit = None

    class print_conf:
        run_types = None
        fold_aggregations = None
        label_aggregations = None
        folds = None
        measures = None
        error_analysis = None
        label_distribution = None
        top_k = 3

    class representation_conf:
        name = None
        dimension = None
        term_list = None
        aggregation = None

    class transform_conf:
        dimension = None
        name = None

    class semantic_conf:
        name = None
        enrichment = None
        weights = None

    class learner_conf:
        name = None

        # clusterers
        num_clusters = None

        # dnns
        hidden_dim = None
        num_layers = None
        sequence_length = None

    class folders_conf:
        run = None
        results = None
        serialization = None
        raw_data = None
        logs = None

    class train_conf:
        epochs = None
        folds = None
        early_stopping_patience = None
        validation_portion = None
        batch_size = None
        sampling_method = None
        sampling_ratios = None

    class misc_conf:
        seed = None
        keys = {}
        csv_separator = ","
        independent_component = None
        allow_deserialization = None
        allow_model_loading = None
        allow_prediction_loading = None

    def __init__(self, conf_file=None):
        "Configuration object constructor"
        self.dataset = Config.dataset_conf()
        self.representation = Config.representation_conf()
        self.transform = Config.transform_conf()
        self.semantic = Config.semantic_conf()
        self.learner = Config.learner_conf()
        self.train = Config.train_conf()
        self.misc = Config.misc_conf()
        self.print = Config.print_conf()
        self.folders = Config.folders_conf()
        if conf_file is not None:
            self.initialize(conf_file)

    def get_pipeline(self):
        return self.pipeline

    def has_data_limit(self):
        return self.dataset.data_limit is not None and any([x is not None for x in self.dataset.data_limit])

    def has_class_limit(self):
        return self.dataset.class_limit is not None

    def has_limit(self):
        return self.has_data_limit() or self.has_class_limit()

    def initialize(self, configuration):
        # read global configuration

        self.read_config(configuration)
        if self.run_id is None:
            self.run_id = utils.datetime_str()
        self.make_directories()
        # copy configuration to run folder
        if type(configuration) is str:
            config_copy = join(self.folders.run, os.path.basename(configuration))
            if not exists(config_copy):
                shutil.copy(configuration, config_copy)

        self.setup_logging()

        # read chains
        self.read_chains(configuration)

    def make_directories(self):
        os.makedirs(self.folders.run, exist_ok=True)
        os.makedirs(self.folders.raw_data, exist_ok=True)
        os.makedirs(self.folders.serialization, exist_ok=True)

    # get option
    def option(self, name):
        if "options" not in self.conf or not self.conf["options"]:
            return False
        if name in self.conf["options"]:
            return self.conf["options"][name]
        return False

    # logging initialization
    def setup_logging(self):
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)7s | %(message)s')

        # console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        lvl = logging._nameToLevel[self.print.log_level.upper()]
        # logger = logging.getLogger(self.logger_name)
        logger = logging.getLogger()
        logger.setLevel(lvl)
        logger.addHandler(handler)

        # file handler
        self.logfile = os.path.join(self.folders.run, "log_{}.log".format(self.run_id))
        fhandler = logging.FileHandler(self.logfile)
        fhandler.setLevel(lvl)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

        self.logger = logger
        return logger

    # logger fetcher
    def get_logger(self):
        return self.logger

    def read_chains(self, input_config):
        self.pipeline = Pipeline()
        if type(input_config) is str:
            try:
                # it's a yaml file
                with open(input_config) as f:
                    self.conf = utils.ordered_load(f, Loader=yaml.SafeLoader)
                    # self.conf = yaml.load(f, Loader=yaml.SafeLoader)
            except yaml.parser.ParserError as p:
                error("Failed to parse input config file: {}".format(input_config))
        elif type(input_config) is dict:
            self.conf = input_config

        need(self.has_value("chains"), "Need chain information")
        for chain in self.conf["chains"]:
            self.pipeline.add_chain(self.read_chain(chain))

    def read_chain(self, chain_name):
        chain_configuration_dict = self.conf["chains"][chain_name]
        # chain_configuration_dict["run_id"] = chain_name
        # make object
        chain_config = self.get_copy()
        # parse the configuration
        read_order = chain_config.read_config(chain_configuration_dict)
        # clear up read order
        return Chain(chain_name, list(chain_configuration_dict.keys()), read_order)

    # read yaml configuration
    def read_config(self, input_config):
        # read yml file
        if type(input_config) is str:
            try:
                # it's a yaml file
                with open(input_config) as f:
                    self.conf = ordered_load(f, Loader=yaml.SafeLoader)
                    # self.conf = yaml.load(f, Loader=yaml.SafeLoader)
            except yaml.parser.ParserError as p:
                error("Failed to parse input config file: {}".format(input_config))
        elif type(input_config) in [dict, OrderedDict]:
            self.conf = input_config

        try:
            fields = list(input_config.keys())
            fields.reverse()
        except:
            fields = [x for x in self.conf.keys() if x != "chains"]
        read_order = []
        while fields:
            field = fields.pop()
            # read dataset options
            # need(self.has_value("dataset"), "Need dataset information")
            if field == "dataset":
                dataset_opts = self.conf["dataset"]
                self.dataset.name = dataset_opts["name"]
                self.dataset.data_limit = self.get_value("data_limit", base=dataset_opts, default=None, expected_type=list)
                self.dataset.class_limit = self.get_value("class_limit", base=dataset_opts, default=None, expected_type=int)
                self.dataset.prepro = self.get_value("prepro", base=dataset_opts, default=None)
                field = self.dataset

            elif field == "link":
                self.link = self.conf["link"]
                field = self.link

            # read representation options
            # need(self.has_value("representation"), "Need representation information")
            elif field == "representation":
                representation_information = self.conf["representation"]
                self.representation.name = representation_information["name"]
                self.representation.aggregation = self.get_value("aggregation", base=representation_information, default=defs.alias.none)
                self.representation.dimension = representation_information["dimension"]
                self.representation.sequence_length = self.get_value("sequence_length", default=1, base=representation_information)
                self.representation.missing_words = self.get_value("unknown_words", default="unk", base=representation_information)
                self.representation.term_list = self.get_value("term_list", base=representation_information)
                self.representation.limit = self.get_value("limit", base=representation_information, default=[])
                field = self.representation

            elif field == "transform":
                transform_opts = self.conf["transform"]
                self.transform.name = transform_opts["name"]
                self.transform.dimension = transform_opts["dimension"]
                field = self.transform

            elif field == "manip":
                manip_opts = self.conf["manip"]
                self.manip.name = manip_opts["name"]
                self.manip.times = self.get_value("times", base=manip_opts, default=None)
                field = self.manip


            elif field == "semantic":
                semantic_opts = self.conf["semantic"]
                self.semantic.name = semantic_opts["name"]
                self.semantic.unit = semantic_opts["unit"]
                self.semantic.enrichment = self.get_value("enrichment", base=semantic_opts, default=None)
                self.semantic.disambiguation = semantic_opts["disambiguation"]
                self.semantic.weights = semantic_opts["weights"]
                self.semantic.limit = self.get_value("limit", base=semantic_opts, default=[], expected_type=list)
                # context file only relevant on semantic embedding disamgibuation
                self.semantic.context_file = self.get_value("context_file", base=semantic_opts)
                self.semantic.context_aggregation = self.get_value("context_aggregation", base=semantic_opts)
                self.semantic.context_threshold = self.get_value("context_threshold", base=semantic_opts)
                self.semantic.spreading_activation = self.get_value("spreading_activation", base=semantic_opts, expected_type=list, default=[])
                field = self.semantic

            # need(self.has_value("learner"), "Need learning information")
            elif field == "learner":
                learner_opts = self.conf["learner"]
                self.learner.name = learner_opts["name"]
                self.learner.hidden_dim = learner_opts["hidden_dim"]
                self.learner.num_layers = learner_opts["layers"]
                self.learner.sequence_length = self.get_value("sequence_length", default=1, base=learner_opts)
                self.learner.num_clusters = self.get_value("num_clusters", default=None, base=learner_opts)
                field = self.learner

            # need(self.has_value("train"), "Need training information")
            elif field == "train":
                training_opts = self.conf["train"]
                self.train.epochs = training_opts["epochs"]
                self.train.folds = self.get_value("folds", default=None, base=training_opts)
                self.train.validation_portion = self.get_value("validation_portion", default=None, base=training_opts)
                self.train.early_stopping_patience = self.get_value("early_stopping_patience", default=None, base=training_opts)
                self.train.batch_size = training_opts["batch_size"]
                self.train.sampling_method = self.get_value("sampling_method", default=None, base=training_opts)
                self.train.sampling_ratios = self.get_value("sampling_ratios", default=None, base=training_opts, expected_type=list)
                field = self.train

            elif field == "folders":
                folder_opts = self.conf["folders"]
                self.folders.run = folder_opts["run"]
                self.folders.results = join(self.folders.run, "results")
                self.folders.serialization = self.get_value("serialization", base=folder_opts, default="serialization")
                self.folders.raw_data = self.get_value("raw_data", base=folder_opts, default="raw_data")
                nltk_data_path = self.get_value("nltk", base=folder_opts, default=os.path.join(self.folders.raw_data, "nltk"))
                # set nltk data folder
                nltk.data.path = [nltk_data_path]

            elif field == "print":
                print_opts = self.conf['print']
                self.print.run_types = self.get_value("run_types", base=print_opts)
                self.print.measures = self.get_value("measures", base=print_opts)
                self.print.label_aggregations = self.get_value("label_aggregations", base=print_opts)
                self.print.folds = self.get_value("folds", base=print_opts, default=False)
                self.print.training_progress = self.get_value("training_progress", base=print_opts, default=True)
                self.print.fold_aggregations = self.get_value("fold_aggregations", base=print_opts)
                self.print.error_analysis = self.get_value("error_analysis", base=print_opts, default=True)
                self.print.top_k = self.get_value("top_k", base=print_opts, default=3, expected_type=int)
                self.print.log_level = self.get_value("log_level", base=print_opts, default="info")
                self.print.label_distribution = self.get_value("label_distribution", base=print_opts, default="logs")
                field = self.print

            elif self.has_value("misc"):
                misc_opts = self.conf["misc"]
                if self.has_value("keys", base=misc_opts):
                    for kname, kvalue in misc_opts['keys'].items():
                        self.misc.keys[kname] = kvalue
                self.misc.independent = self.get_value("independent_component", base=misc_opts, default=False)
                self.misc.allow_deserialization = self.get_value("deserialization_allowed", base=misc_opts, default=True)
                self.misc.allow_prediction_loading = self.get_value("prediction_loading_allowed", base=misc_opts, default=False)
                self.misc.allow_model_loading = self.get_value("model_loading_allowed", base=misc_opts, default=False)
                self.misc.csv_separator = self.get_value("csv_separator", base=misc_opts, default=",")
                self.misc.run_id = self.get_value("run_id", base=misc_opts, default="run_" + utils.datetime_str())

                if not self.has_value("keys", base=misc_opts):
                    self.misc.seed = random.randint(0, 5000)
                    warning("No random seed submitted, randomly generated: {}".format(self.misc.seed))
                else:
                    self.misc.seed = self.get_value("seed", base=misc_opts)
                field = self.misc

            read_order.append(self)

        info("Read configuration for run / chain: {} from the file {}".format(self.run_id, input_config if type(input_config) is str else input_config.keys()))
        return read_order

    def has_transform(self):
        return self.transform.name not in [defs.alias.none, None]

    def has_semantic(self):
        return all([x not in [defs.alias.none, None] for x in [self.semantic.enrichment, self.semantic.name]])

    def is_debug(self):
        return self.conf["log_level"] == "debug"

    def get_train_params(self):
        return self.conf["train"]

    def get_value(self, name, default=None, base=None, expected_type=None):
        if base is None:
            base = self.conf
        value = base[name] if name in base else default
        if expected_type is not None and value is not None:
            if type(value) is not expected_type:
                error("Argument {} got value {} which is of type {}, but {} is required."
                      .format(name, value, type(value), expected_type))
        return value

    def has_value(self, name, base=None):
        if base is None:
            base = self.conf
        return name in base and base[name] is not None and base[name] != defs.alias.none

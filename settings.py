import os
from os.path import join, exists
import logging
import random
import yaml
import utils
import defs
from utils import need, error
import shutil


class Config:
    log_dir = "logs"
    seed_file = "seed.txt"
    # logger_name = "root"
    logger = None
    seed = None
    conf = {}
    run_id = None

    name_value_entries = {}

    class dataset:
        name = None
        prepro = None
        data_limit = [None, None]
        class_limit = None

    class print:
        run_types = None
        stats = None
        aggregations = None
        folds = None
        measures = None

    class representation:
        name = None
        dimension = None
        term_list = None

    class transform:
        dimension = None
        name = None

    class semantic:
        name = None
        enrichment = None
        weights = None

    class learner:
        name = None
        hidden_dim = None
        num_layers = None
        sequence_length = None
        noload = False

        def to_str():
            return "{} {} {} {}".format(Config.learner.name, Config.learner.hidden_dim, Config.learner.num_layers, Config.learner.sequence_length)

    class folders:
        run = None
        results = None
        serialization = None
        logs = None

    class train:
        epochs = None
        folds = None
        early_stopping_patience = None
        validation_portion = None
        batch_size = None

    class misc:
        keys = {}

    def __init__(self, conf_file):
        "Configuration object constructor"
        self.initialize(conf_file)

    def has_data_limit(self):
        return self.dataset.data_limit is not None and any([x is not None for x in self.dataset.data_limit])

    def has_class_limit(self):
        return self.dataset.class_limit is not None and self.dataset.class_limit is not None

    def has_limit(self):
        return self.has_data_limit() or self.has_class_limit()

    def get_run_id(self):
        return self.run_id

    def initialize(self, configuration):
        self.read_config(configuration)
        # copy configuration to run folder
        if not exists(self.folders.run):
            os.makedirs(self.folders.run, exist_ok=True)
        if type(configuration) == str:
            config_copy = join(self.folders.run, os.path.basename(configuration))
            if not exists(config_copy):
                shutil.copy(configuration, config_copy)

        self.setup_logging()
        self.setup_seed()

    # get option
    def option(self, name):
        if "options" not in self.conf or not self.conf["options"]:
            return False
        if name in self.conf["options"]:
            return self.conf["options"][name]
        return False

    # logging initialization
    def setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        # console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        lvl = logging._nameToLevel[self.log_level.upper()]
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

    # random seed initialization
    def setup_seed(self):
        # return already acquired seed
        if self.seed is not None:
            return self.seed

        if os.path.isfile(self.seed_file):
            # read existing from seed file
            with open(self.seed_file) as f:
                seed = f.readlines()[0]
            self.logger.info("Read seed from file {} : {}".format(self.seed_file, seed))
            self.seed = int(seed)
        else:
            # generate new seed
            self.seed = random.randint(0, 5000)
            self.logger.info("Generated seed {}, writing to file {}".format(self.seed, self.seed_file))
            with open(self.seed_file, "w") as f:
                f.write("{}".format(self.seed))
        return self.seed

    # logger fetcher
    def get_logger(self):
        return self.logger

    # seed fetcher
    def get_seed(self):
        return self.seed

    # read yaml configuration
    def read_config(self, input_config):
        # read yml file
        if type(input_config) == str:
            # it's a yaml file
            with open(input_config) as f:
                self.conf = yaml.load(f)
        elif type(input_config) == dict:
            self.conf = input_config

        # setup run id
        if self.explicit_run_id():
            self.run_id = self.conf["run_id"]
        else:
            self.run_id = "run_" + utils.datetime_str()

        # read dataset options
        need(self.has_value("dataset"), "Need dataset information")
        dataset_opts = self.conf["dataset"]
        self.dataset.name = dataset_opts["name"]
        lims = self.get_value("data_limit", base=dataset_opts, default=None)
        if lims is not None:
            lims = utils.aslist(lims)
            if not(all([type(x) == int and x > 0 for x in lims]) and len(lims) in [1, 2]):
                error("Invalid data limits: {}".format(lims))
            self.dataset.data_limit = lims
            if len(lims) == 1:
                self.dataset.data_limit = [lims[0] if type(lims) == list else lims, None]
            self.dataset.class_limit = self.get_value("class_limit", base=dataset_opts, default=None)
        self.dataset.prepro = self.get_value("prepro", base=dataset_opts, default=None)

        # read representation options
        need(self.has_value("representation"), "Need representation information")
        representation_information = self.conf["representation"]
        self.representation.name = representation_information["name"]
        self.representation.aggregation = representation_information["aggregation"] if type(representation_information["aggregation"]) == list else [representation_information["aggregation"]]
        self.representation.dimension = representation_information["dimension"]
        self.representation.sequence_length = self.get_value("sequence_length", default=1, base=representation_information)
        self.representation.missing_words = self.get_value("unknown_words", default="unk", base=representation_information)
        self.representation.term_list = self.get_value("term_list", base=representation_information)
        self.representation.limit = self.get_value("limit", base=representation_information, default=defs.limit.none)

        if self.has_value("transform"):
            transform_opts = self.conf["transform"]
            self.transform.name = transform_opts["name"]
            self.transform.dimension = transform_opts["dimension"]

        if self.has_value("semantic"):
            semantic_opts = self.conf["semantic"]
            self.semantic.name = semantic_opts["name"]
            self.semantic.unit = semantic_opts["unit"]
            self.semantic.enrichment = self.get_value("enrichment", base=semantic_opts, default=None)
            self.semantic.disambiguation = semantic_opts["disambiguation"]
            self.semantic.weights = semantic_opts["weights"]
            self.semantic.limit = self.get_value("limit", base=semantic_opts, default=None, expected_type=list)
            # context file only relevant on semantic embedding disamgibuation
            self.semantic.context_file = self.get_value("context_file", base=semantic_opts)
            self.semantic.context_aggregation = self.get_value("context_aggregation", base=semantic_opts)
            self.semantic.context_threshold = self.get_value("context_threshold", base=semantic_opts)
            self.semantic.spreading_activation = self.get_value("spreading_activation", base=semantic_opts, expected_type=list)

        need(self.has_value("learner"), "Need learner information")
        learner_opts = self.conf["learner"]
        self.learner.name = learner_opts["name"]
        self.learner.hidden_dim = learner_opts["hidden_dim"]
        self.learner.num_layers = learner_opts["layers"]
        self.learner.sequence_length = self.get_value("sequence_length", default=1, base=learner_opts)
        self.learner.no_load = self.get_value("no_load", default=False, base=learner_opts)

        need(self.has_value("train"), "Need training information")
        training_opts = self.conf["train"]
        self.train.epochs = training_opts["epochs"]
        self.train.folds = self.get_value("folds", default=None, base=training_opts)
        self.train.validation_portion = self.get_value("validation_portion", default=None, base=training_opts)
        self.train.early_stopping_patience = self.get_value("early_stopping_patience", default=None, base=training_opts)
        self.train.batch_size = training_opts["batch_size"]

        if self.has_value("folders"):
            folder_opts = self.conf["folders"]
            self.folders.run = folder_opts["run"]
            self.folders.results = join(self.folders.run, "results")
            self.folders.serialization = self.get_value("serialization", base=folder_opts, default="serialization")
            self.folders.raw_data = self.get_value("raw_data", base=folder_opts, default="raw_data")

        if self.has_value("print"):
            print_opts = self.conf['print']
            self.print.run_types = self.get_value("run_types", base=print_opts)
            self.print.measures = self.get_value("measures", base=print_opts)
            self.print.aggregations = self.get_value("aggregations", base=print_opts)
            self.print.folds = self.get_value("folds", base=print_opts, default=False)
            self.print.training_progress = self.get_value("training_progress", base=print_opts, default=False)
            self.print.stats = self.get_value("stats", base=print_opts)

        if self.has_value("misc"):
            if self.has_value("keys", base=self.conf["misc"]):
                for kname, kvalue in self.conf["misc"]['keys'].items():
                    self.misc.keys[kname] = kvalue

        self.log_level = self.get_value("log_level", default="info")

        print("Read configuration for run {} from {}".format(self.run_id, input_config))

    def has_transform(self):
        return self.transform.name not in [defs.alias.none, None]

    def has_semantic(self):
        return all([x not in [defs.alias.none, None] for x in [self.semantic.enrichment, self.semantic.name]])

    def is_debug(self):
        return self.conf["log_level"] == "debug"

    def get_train_params(self):
        return self.conf["train"]

    def explicit_run_id(self):
        return self.has_value("run_id")

    def get_value(self, name, default=None, base=None, expected_type=None):
        if base is None:
            base = self.conf
        value = base[name] if name in base else default
        if expected_type is not None and value is not None:
            if type(value) != expected_type:
                error("Argument {} got value {} which is of type {}, but {} is required."
                      .format(name, value, type(value), expected_type))
        return value

    def has_value(self, name, base=None):
        if base is None:
            base = self.conf
        return name in base and base[name] is not None and base[name] != defs.alias.none

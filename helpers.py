import os
import logging
import random
import yaml
import utils
from utils import need


class Config:
    log_dir = "logs"
    seed_file = "seed.txt"
    # logger_name = "root"
    logger = None
    seed = None
    conf = {}
    run_id = None

    class dataset:
        name = None
        limit = None

    class embedding:
        dimension = None

    class semantic:
        name = None
        enrichment = None
        weights = None

    class learner:
        name = None
        hidden_dim = None
        num_layers = None
        sequence_length = None

    class folders:
        results = None
        serialization = None
        logs = None

    class train:
        epochs = None
        folds = None
        early_stopping_patience = None
        validation_portion = None
        batch_size = None

    def get_run_id(self):
        return self.run_id

    def initialize(self, config_file):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.read_config(config_file)
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
        logfile = os.path.join(self.log_dir, "log_{}.txt".format(self.run_id))
        fhandler = logging.FileHandler(logfile)
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
    def read_config(self, yaml_file):
        # read yml file
        with open(yaml_file) as f:
            self.conf = yaml.load(f)

        # setup run id
        if self.explicit_run_id():
            self.run_id = self.conf["run_id"]
        else:
            self.run_id = "run_" + utils.datetime_str()

        # read dataset options
        need(self.has_value("dataset"), "Need dataset information")
        dataset_opts = self.conf["dataset"]
        self.dataset.name = dataset_opts["name"]
        self.dataset.limit = self.get_value("limit", base = dataset_opts)
        # read embedding options
        need(self.has_value("embedding"), "Need embedding information")
        embedding_opts = self.conf["embedding"]
        self.embedding.name = embedding_opts["name"]
        self.embedding.aggregation = embedding_opts["aggregation"] if type(embedding_opts["aggregation"]) == list else [embedding_opts["aggregation"]]
        self.embedding.dimension = embedding_opts["dimension"]
        self.embedding.sequence_length = embedding_opts["sequence_length"]

        if self.has_value("semantic"):
            semantic_opts = self.conf["semantic"]
            self.semantic.name = semantic_opts["name"]
            self.semantic.unit = semantic_opts["unit"]
            self.semantic.enrichment = semantic_opts["enrichment"]
            self.semantic.disambiguation = semantic_opts["disambiguation"]
            self.semantic.weights = semantic_opts["weights"]
            self.semantic.frequency_threshold = semantic_opts["frequency_threshold"]
            # context file only relevant on semantic embedding disamgibuation
            self.semantic.context_file = self.get_value(semantic_opts["context_file"])
            self.semantic.context_aggregation = self.get_value(semantic_opts["context_aggregation"])
            self.semantic.context_threshold = self.get_value(semantic_opts["context_freq_threshold"])

        need(self.has_value("learner"), "Need learner information")
        learner_opts = self.conf["learner"]
        self.learner.name = learner_opts["name"]
        self.learner.hidden_dim = learner_opts["hidden_dim"]
        self.learner.num_layers = learner_opts["layers"]
        self.learner.sequence_length = self.get_value("sequence_length", default=None, base = learner_opts)

        need(self.has_value("train"), "Need training information")
        training_opts = self.conf["train"]
        self.train.epochs = training_opts["epochs"]
        self.train.folds = training_opts["folds"]
        self.train.validation_portion = self.get_value("validation_portion", default = 0.1, base=training_opts)
        self.train.early_stopping_patience = self.get_value("early_stopping_patience", default=None, base=training_opts)
        self.train.batch_size = training_opts["batch_size"]

        if self.has_value("folders"):
            folder_opts = self.conf["folders"]
            self.folders.logs = self.get_value("logs", base=folder_opts, default="logs")
            self.folders.results = self.get_value("results", base=folder_opts, default="results")
            self.folders.serialization = self.get_value("serialization", base=folder_opts, default="serialization")
            self.folders.embeddings = self.get_value("embeddings", base=folder_opts, default="embeddings")
            self.folders.semantic = self.get_value("semantic", base=folder_opts, default="semantic")

        self.log_level = self.get_value("log_level", default="info")

        print("Read configuration for run {} from {}".format(self.run_id, yaml_file))

    # configuration entry getters
    def get_serialization_dir(self):
        return self.conf["serialization_dir"]

    def get_dataset(self):
        return self.conf["dataset"]

    def has_enrichment(self):
        return self.has_value("semantic")

    def is_debug(self):
        return self.conf["log_level"] == "debug"

    def get_semantic_resource(self):
        return self.conf["semantic_resource"]

    def get_semantic_disambiguation(self):
        return self.conf["semantic_disambiguation"]

    def get_semantic_word_context(self):
        return self.conf["semantic_word_context_file"]

    def get_semantic_word_aggregation(self):
        return self.conf["semantic_embedding_aggregation"]

    def get_semantic_freq_threshold(self):
        return self.get_value("semantic_freq_threshold")

    def get_semantic_weights(self):
        return self.conf["semantic_weights"]

    def get_enrichment(self):
        return self.get_value("enrichment")

    def get_learner(self):
        return self.conf["learner"]

    def get_embedding(self):
        return self.get_value("embedding")

    def get_train_params(self):
        return self.conf["train"]

    def explicit_run_id(self):
        return self.has_value("run_id")

    def get_value(self, name, default=None, base=None):
        if base is None:
            base = self.conf
        return base[name] if name in base else default

    def has_value(self, name):
        return name in self.conf and self.conf[name]

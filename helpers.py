import os
import time
import logging
import random
import yaml
import utils



class Config:
    log_dir = "logs"
    seed_file = "seed.txt"
    # logger_name = "root"
    logger = None
    seed = None
    conf = {}
    run_id = None

    def get_run_id(self):
        return self.run_id

    def initialize(self, config_file):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.run_id = "run_" + utils.datetime_str()
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
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        # console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        log_level = self.conf["log_level"]
        if log_level == "info":
            log_level = logging.INFO
        elif log_level == "debug":
            log_level = logging.DEBUG

        # logger = logging.getLogger(self.logger_name)
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(handler)

        # file handler
        logfile = os.path.join(self.log_dir,"log_{}.txt".format(self.run_id))
        fhandler = logging.FileHandler(logfile)
        fhandler.setLevel(log_level)
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
            self.seed = seed
        else:
            # generate new seed
            self.seed = random.randint(0,5000)
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
        print("Reading configuration for run {} from {}".format(self.run_id, yaml_file))
        with open(yaml_file) as f:
            self.conf = yaml.load(f)

    # configuration entry getters
    def get_serialization_dir(self):
        return self.conf["serialization_dir"]

    def get_dataset(self):
        return self.conf["dataset"]

    def get_results_folder(self):
        return self.conf["results_folder"]

    def get_semantic_resource(self):
        return self.conf["semantic_resource"]

    def get_learner(self):
        return self.conf["learner"]

    def get_aggregation(self):
        return self.conf["aggregation"]

    def get_embedding(self):
        return self.conf["embedding"]

    def get_batch_size(self):
        return self.conf["batch_size"]

    def get_train_params(self):
        return self.conf["train"]


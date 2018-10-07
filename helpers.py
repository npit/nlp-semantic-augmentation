import os
import logging
import random
import yaml



class Config:
    seed_file = "seed.txt"
    # logger_name = "root"
    logger = None
    seed = None
    conf = {}

    def initialize(self, config_file):
        self.setup_logging()
        self.setup_seed()
        self.read_config(config_file)

    # logging initialization
    def setup_logging(self):
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # logger = logging.getLogger(self.logger_name)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
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
        self.logger.info("Reading configuration from {}".format(yaml_file))
        with open(yaml_file) as f:
            self.conf = yaml.load(f)

    # configuration entry getters
    def get_dataset(self):
        return self.conf["dataset"]

    def get_semantic_resource(self):
        return self.conf["semantic_resource"]

    def get_classifier(self):
        return self.conf["classifier"]

    def get_embedding(self):
        return self.conf["embedding"]

def error(msg):
    logger = logging.getLogger()
    logger.error(msg)
    raise Exception(msg)

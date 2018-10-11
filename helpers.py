import os
import time
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


def error(msg):
    logger = logging.getLogger()
    logger.error(msg)
    raise Exception(msg)

class Timer:
    times = []

def tic():
    Timer.times.append(time.time())

def toc(msg):
    logger = logging.getLogger()
    elapsed = time.time() - Timer.times.pop()
    # convert to smhd
    minutes = elapsed // 60
    elapsed -= minutes * 60
    seconds = elapsed

    hours = minutes // 60
    minutes -= hours * 60

    days = hours // 24
    hours -= days*24

    elapsed = ""
    names = ["days", "hours", "minutes", "seconds"]
    values = [days, hours, minutes, seconds]
    elapsed = "".join([ elapsed + "{:.3f} {} ".format(x, n) for (n,x) in zip(names, values) if x > 0])
    logger.info("{} took {}.".format(msg, elapsed))

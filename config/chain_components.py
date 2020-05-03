"""Module for defining configuration for chain components
"""
import defs
from config.config import Configuration
from utils import as_list


class manip_conf(Configuration):
    conf_key_name = "manip"
    name = None
    times = None

    def __init__(self, config):
        """Constructor for the manipulator component configuration"""
        super().__init__(config)
        self.name = config["name"]
        self.times = self.get_value("times", base=config, default=None)


class dataset_conf(Configuration):
    conf_key_name = "dataset"
    name = None
    prepro = None
    data_limit = [None, None]
    class_limit = None

    def __init__(self, conf):
        """Constructor for the dataset component configuration"""
        super().__init__(conf)
        if conf is None:
            return
        self.name = conf["name"]
        self.data_limit = self.get_value("data_limit", default=None, expected_type=list)
        self.class_limit = self.get_value("class_limit", default=None, expected_type=int)
        self.prepro = self.get_value("prepro", default=None)

    def has_data_limit(self):
        return self.data_limit is not None and any([x is not None for x in self.data_limit])

    def has_class_limit(self):
        return self.class_limit is not None

    def has_limit(self):
        return self.has_data_limit() or self.has_class_limit()


class representation_conf(Configuration):
    conf_key_name = "representation"
    name = None
    dimension = None
    term_list = None
    aggregation = None
    train = None

    def __init__(self, config):
        """Constructor for the representation configuration"""
        super().__init__(config)
        if config is None:
            return
        self.name = config["name"]
        self.aggregation = self.get_value("aggregation", base=config, default=defs.alias.none)
        self.dimension = config["dimension"]
        self.sequence_length = self.get_value("sequence_length", default=1, base=config)
        self.missing_words = self.get_value("unknown_words", default="unk", base=config)
        self.term_list = self.get_value("term_list", base=config)
        self.limit = self.get_value("limit", base=config, default=[])
        self.train = self.get_value("train", base=config, default=False)


class transform_conf(Configuration):
    conf_key_name = "transform"
    dimension = None
    name = None

    def __init__(self, conf):
        """Constructor for the transform configuration"""
        super().__init__(conf)
        if conf is None:
            return
        self.name = self.conf["name"]
        self.dimension = self.conf["dimension"]


class semantic_conf(Configuration):
    conf_key_name = "semantic"
    name = None
    enrichment = None
    weights = None

    def __init__(self, config):
        """Constructor for the semantic configuration"""
        super().__init__(config)
        if config is None:
            return
        self.name = config["name"]
        self.unit = config["unit"]
        self.enrichment = self.get_value("enrichment", base=config, default=None)
        self.disambiguation = config["disambiguation"]
        self.weights = config["weights"]
        self.limit = self.get_value("limit", base=config, default=[], expected_type=list)
        # context file only relevant on semantic embedding disamgibuation
        self.context_file = self.get_value("context_file", base=config)
        self.context_aggregation = self.get_value("context_aggregation", base=config)
        self.context_threshold = self.get_value("context_threshold", base=config)
        self.spreading_activation = self.get_value("spreading_activation", base=config, expected_type=list, default=[])


class learner_conf(Configuration):
    conf_key_name = "learner"
    name = None

    # clusterers
    num_clusters = None

    # dnns
    hidden_dim = None
    num_layers = None
    sequence_length = None

    # training configuration
    class train:
        epochs = None
        folds = None
        early_stopping_patience = None
        validation_portion = None
        batch_size = None
        sampling_method = None
        sampling_ratios = None
        optimizer = None

    def __init__(self, config):
        """Constructor for the learner configuration"""
        super().__init__(config)
        if config is None:
            return
        self.name = config["name"]
        if "hidden_dim" in config:
            self.hidden_dim = config["hidden_dim"]
        if "layers" in config:
            self.num_layers = config["layers"]
        try:
            self.sequence_length = self.get_value("sequence_length", default=1, base=config)
        except KeyError:
            self.sequence_length = 1
        if "num_clusters" in config:
            self.num_clusters = self.get_value("num_clusters", default=None, base=config)
        # training parameters
        self.train = learner_conf.train()
        config = config["train"]
        self.train.epochs = config["epochs"]
        self.train.train_embedding = self.get_value("train_embedding", default=False, base=config)
        self.train.optimizer = self.get_value("optimizer", default="sgd", base=config)
        self.train.folds = self.get_value("folds", default=None, base=config)
        self.train.validation_portion = self.get_value("validation_portion", default=None, base=config)
        self.train.early_stopping_patience = self.get_value("early_stopping_patience", default=None, base=config)
        self.train.batch_size = config["batch_size"]
        self.train.sampling_method = self.get_value("sampling_method", default=None, base=config)
        self.train.sampling_ratios = self.get_value("sampling_ratios", default=None, base=config, expected_type=list)


class link_conf(Configuration):
    """Dummy configuration, linking chains with each other"""
    conf_key_name = "link"

    def __init__(self, config):
        super().__init__(config)
        if config is None:
            return
        # pass the linking value
        self.links = as_list(config)

    def get_links(self):
        return self.links


chain_component_classes = [manip_conf, dataset_conf, representation_conf, transform_conf, semantic_conf, learner_conf, link_conf]

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
        self.extract_pos = self.get_value("extract_pos", default=False)
        self.filter_stopwords = self.get_value("filter_stopwords", default=True)
        self.remove_digits = self.get_value("remove_digits", default=False)

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

    def __init__(self, config):
        """Constructor for the representation configuration"""
        super().__init__(config)
        if config is None:
            return
        self.name = config["name"]
        self.aggregation = self.get_value("aggregation", base=config, default=defs.alias.none)
        self.dimension = self.get_value("dimension", base=config, default=None)
        self.sequence_length = self.get_value("sequence_length", default=1, base=config)
        self.missing_words = self.get_value("unknown_words", default="unk", base=config)
        self.term_list = self.get_value("term_list", base=config)
        self.ngram_range = self.get_value("ngram_range", base=config, default=None)
        self.limit = self.get_value("limit", base=config, default=[])


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
        # self.unit = config["unit"]
        self.enrichment = self.get_value("enrichment", base=config, default=None)
        self.disambiguation = self.get_value("disambiguation", default=defs.disam.first)
        self.weights = self.get_value("weights", default=defs.weights.bag)
        self.ngram_range = self.get_value("weights", default=None)
        self.max_terms = self.get_value("max_terms", base=config, default=[], expected_type=int)
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

    use_gpu = True

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
        lr_scheduler = None
        do_test = True

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
        self.use_gpu = self.get_value("use_gpu", default=True, base=config)
        self.do_test = self.get_value("do_test", default=True, base=config)
        self.allow_prediction_loading = self.get_value("allow_prediction_loading", default=False, base=config)

        # training parameters
        self.train = learner_conf.train()
        config = config["train"]
        self.train.epochs = config["epochs"]
        self.train.train_embedding = self.get_value("train_embedding", default=False, base=config)
        self.train.optimizer = self.get_value("optimizer", default="sgd", base=config)
        self.train.lr_scheduler = self.get_value("lr_scheduler", base=config)
        self.train.base_lr = self.get_value("base_lr", base=config, default=0.01)
        self.train.folds = self.get_value("folds", default=None, base=config)
        self.train.validation_portion = self.get_value("validation_portion", default=None, base=config)
        self.train.early_stopping_patience = self.get_value("early_stopping_patience", default=None, base=config)
        self.train.batch_size = config["batch_size"]
        self.train.sampling_method = self.get_value("sampling_method", default=None, base=config)
        self.train.sampling_ratios = self.get_value("sampling_ratios", default=None, base=config, expected_type=list)
        self.save_interval =  self.get_value("save_interval", default=1, base=config)


class trigger_receptor(Configuration):
    conf_key_name = "triggered"

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

class evaluator_conf(Configuration):
    """Component for evaluating"""
    conf_key_name = "evaluator"

    def __init__(self, config):
        super().__init__(config)
        if config is None:
            return
        self.baselines = self.get_value("baselines", base=config)
        self.averages = self.get_value("averages", base=config, default=True)
        self.top_k = self.get_value("top_k", base=config, default=3, expected_type=int)
        self.iter_aggregations = self.get_value("iter_aggregations", base=config)
        self.label_aggregations = self.get_value("label_aggregations", base=config)
        self.measures = self.get_value("measures", base=config)
        self.label_distribution = self.get_value("show_label_distributions", base=config, default="logs")

class endpoint_conf(Configuration):
    conf_key_name = "endpoint"
    def __init__(self, config):
        super().__init__(config)
        if config is None:
            return
        self.name = self.get_value("name", default="io")
        self.url = self.get_value("url", default="localhost")
        self.port = self.get_value("port", default="9999")
        self.endpoint_name = self.get_value("endpoint_name", default="smaug")

chain_component_classes = [manip_conf, dataset_conf, representation_conf, transform_conf, semantic_conf, learner_conf, link_conf, evaluator_conf, endpoint_conf, trigger_receptor]

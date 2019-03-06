from utils import error, info
import numpy as np
from os.path import join, exists
from evaluator import Evaluator


class Learner:

    save_dir = "models"
    folds = None
    fold_index = 0
    evaluator = None

    def __init__(self):
        """Generic learner constructor
        """
        # initialize evaluation
        self.evaluator = Evaluator(self.config)
        pass

    def make(self, representation, dataset):

        self.verbosity = 1 if self.config.print.training_progress else 0
        # get data and labels
        self.train, self.test = representation.get_data()
        self.train_labels, self.test_labels = [x for x in dataset.get_targets()]
        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(zero_samples_idx))

        # get many handy variables
        self.do_multilabel = dataset.is_multilabel()
        self.num_labels = dataset.get_num_labels()
        self.num_train, self.num_test, self.num_train_labels, self.num_test_labels = \
            list(map(len, [self.train, self.test, self.train_labels, self.test_labels]))
        self.evaluator.set_labels(self.test_labels, self.num_labels)
        self.input_dim = representation.get_dimension()
        self.forbid_load = self.config.learner.no_load
        self.sequence_length = self.config.learner.sequence_length
        self.results_folder = self.config.folders.results
        self.models_folder = join(self.results_folder, "models")
        self.epochs = self.config.train.epochs
        self.folds = self.config.train.folds
        self.validation_portion = self.config.train.validation_portion
        self.do_folds = self.folds and self.folds > 1
        self.do_validate_portion = self.validation_portion is not None and self.validation_portion > 0.0
        self.validation_exists = self.do_folds or self.do_validate_portion
        self.early_stopping_patience = self.config.train.early_stopping_patience

        self.seed = self.config.get_seed()
        np.random.seed(self.seed)

        self.batch_size = self.config.train.batch_size
        info("Learner data/labels: train: {} test: {}".format(self.train.shape, self.test.shape))

        # sanity checks
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(self.folds, self.validation_portion))

        # measure sanity checks
        self.evaluator.check_sanity(self.do_multilabel)

    def is_already_completed(self):
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        if exists(predictions_file):
            info("Reading existing predictions: {}".format(predictions_file))
            return read_pickled(predictions_file)
        return None

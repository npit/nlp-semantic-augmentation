from os import makedirs
from os.path import dirname

import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     StratifiedShuffleSplit)

from bundle.datatypes import Labels, Vectors
from defs import roles
from learning.learner import Learner
from learning.sampling import Sampler
from learning.validation import ValidationSetting
from utils import (count_label_occurences, error, info, is_multilabel, tictoc,
                   write_pickled)


class SupervisedLearner(Learner):
    train_labels, test_labels = None, None

    def __init__(self):
        Learner.__init__(self, consumes=[Vectors.name, Labels.name])

    def count_samples(self):
        """Sample counter that includes the label samples"""
        super().count_samples()
        self.num_train_labels, self.num_test_labels = map(len, [self.train_labels, self.test_labels])

    def make(self):
        super().make()
        self.do_multilabel = is_multilabel(self.train_labels)
        label_counts = count_label_occurences(self.train_labels)
        self.num_labels = len(label_counts)

    def attach_evaluator(self):
        super().attach_evaluator()
        # make sure the validation setting makes sense prior to configuring the evaluator
        if len(self.test_labels) == 0:
            # no test labels; better be some validation
            if not (self.validation_exists and self.validation.use_for_testing):
                error("No test data nor validation setting defined. Cannote evaluate run.")
        # add label-related information to the evaluator
        self.evaluator.set_labelling(self.train_labels, self.num_labels, self.do_multilabel, self.test_labels)
        if self.validation_exists and self.validation.use_for_testing:
            reference_labels = self.train_labels
        else:
            reference_labels = self.test_labels
        self.evaluator.majority_label = count_label_occurences(reference_labels)[0][0]

    def check_sanity(self):
        super().check_sanity()
        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(zero_samples_idx))

    def configure_validation_setting(self):
        self.validation = ValidationSetting(self.folds, self.validation_portion, self.test_data_available(), use_labels=True, do_multilabel=self.do_multilabel)
        self.validation.assign_data(self.embeddings, train_index=self.train_embedding_index, train_labels=self.train_labels, test_labels=self.test_labels, test_index=self.test_embedding_index)

    def configure_sampling(self):
        """Over/sub sampling"""
        if self.do_sampling:
            if type(self.sampling_ratios[0]) is not list:
                self.sampling_ratios = [self.sampling_ratios]
            freqs = count_label_occurences(
                [x for y in self.sampling_ratios for x in y[:2]])
            max_label_constraint_participation = max(freqs, key=lambda x: x[1])
            if self.num_labels > 2 and max_label_constraint_participation[1] > 1:
                error("Sampling should be applied on binary classification or constraining ratio should not be overlapping")

    def show_train_statistics(self, train_labels, val_labels):
        msg = "Training label distribution for validation setting: " + self.validation.get_current_descr()
        self.evaluator.show_label_distribution(train_labels, message=msg)
        if val_labels is not None and val_labels.size > 0:
            msg = "Validation label distribution for validation setting: " + self.validation.get_current_descr()
            self.evaluator.show_label_distribution(val_labels, message=msg)

    def acquire_trained_model(self):
        """Trains the learning model or load an existing instance from a persisted file."""
        with tictoc("Training run [{}] on {} training and {} val data.".format(self.validation, self.num_train, len(self.val_index) if self.val_index is not None else "[none]")):
            model = None
            # check if a trained model already exists
            if self.allow_model_loading:
                model = self.load_model()
            if not model:
                model = self.train_model()
                # create directories
                makedirs(self.models_folder, exist_ok=True)
                self.save_model(model)
            else:
                info("Skipping training due to existing model successfully loaded.")
        return model

    def get_training_inputs(self):
        """Retrieve required data for training a supervised model"""
        return (*super().get_training_inputs(), self.train_labels, self.val_labels)

    def assign_current_run_data(self, iteration_index, trainval):
        """Sets data for the current run"""
        # let learner handle indexes to data
        super().assign_current_run_data(iteration_index, trainval)
        # get label containers and print statistics
        self.train_labels, self.val_labels, self.test_labels = self.validation.get_run_labels(iteration_index, trainval)
        self.show_train_statistics(self.train_labels, self.val_labels)
        # update indexes to label of the evaluator: the trainval indexes
        train_idx, val_idx = trainval
        test_label_index = np.arange(len(self.test_labels)) if not self.validation.use_for_testing else val_idx
        self.evaluator.update_reference_labels(train_idx, test_label_index)

    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):

        if self.do_folds:
            # stratified fold splitting
            info("Training {} with input data: {} samples, {} labels, on {} stratified folds"
                .format(self.name, self.num_train, self.num_train_labels, self.folds))
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
            else:
                splitter = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        if self.do_validate_portion:
            info("Splitting {self.name} with input data: {self.num_train} samples, {self.num_train_labels} labels, on a {self.validation_portion} validation portion")
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_portion, random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(splitter.split(np.zeros(self.num_train_labels), self.train_labels))
        return splits

    def process_component_inputs(self):
        """Component processing for label-related data"""
        super().process_component_inputs()
        error("{} needs label information.".format(self.component_name), not self.inputs.has_labels())
        self.train_labels = self.inputs.get_labels(single=True, role=roles.train)
        self.test_labels = self.inputs.get_labels(single=True, role=roles.test)
        if len(self.train_embedding_index) != len(self.train_labels):
            error(f"Train data-label instance number mismatch: {len(self.train_embedding_index)} data and {len(self.train_labels)}")
        if len(self.test_embedding_index) != len(self.test_labels):
            error(f"Test data-label instance number mismatch: {len(self.test_embedding_index)} data and {len(self.test_labels)}")
        self.multilabel_input = self.inputs.get_labels(single=True).multilabel

    def load_existing_predictions(self):
        """Loader function for existing, already computed predictions. Checks for label matching."""
        # get predictions and instance indexes they correspond to
        existing_predictions, existing_instance_indexes = super().load_existing_predictions()
        # also check labels
        if existing_predictions is not None:
            existing_test_labels = self.validation.get_test_labels(self.test_instance_indexes)
            if not np.all(np.equal(existing_test_labels, self._test_labels)):
                error("Different instance labels loaded than the ones generated.")
        return existing_predictions, existing_instance_indexes

    # def conclude_validation_iteration(self):
    #     """Perform concluding actions for a single validation loop"""
    #     super().conclude_validation_iteration()
    #     if self.validation_exists and self.validation.use_for_testing:
    #         self.test, self.test_labels, self.test_instance_indexes = [], [], None

    def conclude_traintest(self):
        """Perform concuding actions for the entire traintest loop"""
        super().conclude_traintest()
        # show label distribution, if desired
        if self.config.print.label_distribution:
            self.evaluator.show_reference_label_distribution()

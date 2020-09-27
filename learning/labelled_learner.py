from os import makedirs
from os.path import dirname

import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split

from bundle.datatypes import *
from bundle.datausages import *
from defs import roles
from learning.supervised_learner import SupervisedLearner
from learning.sampling import Sampler
from learning.validation.validation import ValidationSetting
from utils import (count_label_occurences, error, info, is_multilabel, tictoc, write_pickled, all_labels_have_samples, one_hot)


class LabelledLearner(SupervisedLearner):
    """Class where the ground truth for each sample is numeric labels"""
    train_labels, test_labels = None, None

    def __init__(self):
        SupervisedLearner.__init__(self, consumes=[Numeric.name, Labels.name])

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
        self.evaluator.set_labelling(self.train_labels, self.labelset, self.do_multilabel, self.test_labels)
        if self.validation_exists and self.validation.use_for_testing:
            reference_labels = self.train_labels
        else:
            reference_labels = self.test_labels
        self.evaluator.majority_label = count_label_occurences(reference_labels)[0][0]

    def check_sanity(self):
        super().check_sanity()
        if not self.do_multilabel:
            # need at least one sample per class
            zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
            if np.any(zero_samples_idx):
                error("No training samples for class index {}".format(zero_samples_idx))
        else:
            ok, nosamples = all_labels_have_samples(self.train_labels, self.labelset)
            if not ok:
                error(f"No samples for label(s): {nosamples}")

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
        if val_labels is not None and len(val_labels) > 0:
            msg = "Validation label distribution for validation setting: " + self.validation.get_current_descr()
            self.evaluator.show_label_distribution(val_labels, message=msg)

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

    def stratified_mutltilabel_split(self):
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[self.validation_portion, 1.0-self.validation_portion])
        labels = one_hot(self.train_labels, self.num_labels, self.do_multilabel)
        train_indexes, test_indexes = next(stratifier.split(np.zeros(len(self.train_labels)), labels))
        return [(train_indexes, test_indexes)]

    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):
        if self.do_folds:
            # stratified fold splitting
            info("Training {} with input data: {} samples, {} labels, on {} stratified folds"
                .format(self.name, self.num_train, self.num_train_labels, self.folds))
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                # splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
                splitter = IterativeStratification(self.folds, order=1)
                oh_labels = one_hot(self.train_labels, self.num_labels, is_multilabel=True)
                return list(splitter.split(np.zeros(self.num_train_labels), oh_labels))
            else:
                splitter = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        elif self.do_validate_portion:
            info(f"Splitting {self.name} with input data: {self.num_train} samples, {self.num_train_labels} labels, on a {self.validation_portion} validation portion")
            if self.do_multilabel:
                # splitter = ShuffleSplit(n_splits=1, test_size=self.validation_portion, random_state=self.seed)

                # splitter = lambda X, y: iterative_train_test_split(X, y, test_size=self.validation_portion)
                # iterative_train_test_split(np.zeros(self.num_train_labels), self.train_labels,test_size=self.validation_portion)
                return self.stratified_mutltilabel_split()
            else:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_portion, random_state=self.seed)

        # generate splits
        splits = list(splitter.split(np.zeros(self.num_train_labels), self.train_labels))
        return splits


    def process_ground_truth_input(self):
        labels = self.data_pool.request_data(Numeric.name, Labels, usage_matching="subset", client=self.name, on_error_message=f"{self.name} learner needs label information.")
        indices = labels.get_usage(Indices.name)
        labels_info = labels.get_usage(Labels.name)
        train_idx, test_idx = indices.get_train_test()
        # TODO fix labels as embeddings
        self.train_labels =  labels.data.instances[0]
        self.test_labels = labels.data.instances[1]
        self.labelset, self.multilabel_input = labels_info.labelset, labels_info.multilabel
        self.num_labels = len(self.labelset)

    def get_ground_truth(self):
        """GT retrieval"""
        return self.train_labels, self.val_labels

    def load_existing_predictions(self):
        """Loader function for existing, already computed predictions. Checks for label matching."""
        # get predictions and instance indexes they correspond to
        existing_predictions, existing_instance_indexes = super().load_existing_predictions()
        # also check labels
        if existing_predictions is not None:
            existing_test_labels = self.validation.get_test_labels(self.test_instance_indexes)
            if not np.all(np.equal(existing_test_labels, self.test_labels)):
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

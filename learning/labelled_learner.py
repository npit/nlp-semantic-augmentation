from os import makedirs
from os.path import dirname

import numpy as np

from bundle.datatypes import *
from bundle.datausages import *
from defs import roles
from learning.supervised_learner import SupervisedLearner
from learning.sampling import Sampler
from learning.validation.validation import ValidationSetting
from utils import (count_occurences, error, info, is_multilabel, tictoc, write_pickled, all_labels_have_samples, one_hot, read_pickled, warning)


class LabelledLearner(SupervisedLearner):
    """Class for supervised learners where the ground truth for each sample are distinct numeric labels"""
    labels_info = None

    def __init__(self):
        SupervisedLearner.__init__(self, consumes=[Numeric.name, Labels.name])

    def count_samples(self):
        """Sample counter that includes the label samples"""
        super().count_samples()
        self.num_train_labels, self.num_test_labels = map(len, [self.train_embedding_index, self.test_embedding_index])

    # def make(self):
    #     super().make()
    #     self.do_multilabel = is_multilabel(self.train_labels)
    #     label_counts = count_label_occurences(self.train_labels)
    #     self.num_labels = len(label_counts)

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
        # need at least one sample per class
        existing_labels = set(np.concatenate(self.targets.instances))
        missing_labels = [x for x in self.labelset if x not in existing_labels]
        error("No training samples for label(s): {missing_labels}", len(missing_labels) > 0)

    def configure_validation_setting(self):
        self.validation = ValidationSetting(self.config, self.train_embedding_index, self.test_embedding_index,
        self.targets, self.labels_info, self.folds, self.validation_portion, self.seed)
        # self.validation.assign_data(self.embeddings, train_index=self.train_embedding_index, labels=self.targets, test_index=self.test_embedding_index)

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

    def process_ground_truth_input(self):
        super().process_ground_truth_input()
        # labels = self.data_pool.request_data(Numeric.name, Labels, usage_matching="subset", client=self.name, on_error_message=f"{self.name} learner needs label information.")
        # indices = labels.get_usage(Indices.name)
        # train_idx, test_idx = indices.get_train_test()
        # # TODO fix labels as embeddings
        if self.targets_data:
            self.labels_info = self.targets_data.get_usage(Labels.name)
            error(f"Learner {self.name} requires numeric label information.", self.labels_info is None)
            self.process_label_information(self.labels_info)

        # self.labels = labels.data
        # self.train_labels =  labels.data.get_slice(train_idx)
        # self.test_labels = labels.data.get_slice(test_idx)
        # self.labelset, self.multilabel_input = labels_info.labelset, labels_info.multilabel
        # self.num_labels = len(self.labelset)


    def get_ground_truth(self):
        # convert to labels
        gt = super().get_ground_truth().instances
        gt = np.concatenate(gt)
        return gt
    # def conclude_validation_iteration(self):
    #     """Perform concluding actions for a single validation loop"""
    #     super().conclude_validation_iteration()
    #     if self.validation_exists and self.validation.use_for_testing:
    #         self.test, self.test_labels, self.test_instance_indexes = [], [], None

    def conclude_traintest(self):
        """Perform concuding actions for the entire traintest loop"""
        super().conclude_traintest()
        # show label distribution, if desired
        pass
        # if self.config.print.label_distribution:
        #     self.evaluator.show_reference_label_distribution()


    def save_model_wrapper(self):
        # get learner wrapper info
        path = self.get_current_model_path() + ".wrapper"
        write_pickled(path, self.labels_info)

    def load_model_wrapper(self):
        # get learner wrapper info
        try:
            path = self.get_current_model_path() + ".wrapper"
            data = read_pickled(path, msg=f"{self.name} metadata")
            return self.process_label_information(data)
        except FileNotFoundError as ex:
            return False

    def clear_model_wrapper(self):
        self.labels_info, self.labelset, self.do_multilabel = None, None, None

    def process_label_information(self, data):
        labelset = data.labelset
        multi = data.multilabel
        num = len(labelset)

        if self.labels_info is not None:
            # check compatibility with loaded inputs
            if num != len(self.labels_info.labelset):
                warning(f"Got {len(self.labels_info.labelset)} labels from inputs but processed {num}")
                return False
            if self.labels_info.labelset != labelset:
                warning(f"Got labelset: {self.labels_info.labelset} from inputs but processed labelset: {labelset}")
                return False
            if self.labels_info.multilabel != multi:
                warning(f"Got multilabel setting: {self.labels_info.do_multilabel} from inputs but processed: {multi}")
                return False

        self.labels_info = data
        self.labelset, self.do_multilabel, self.num_labels = labelset, multi, num
        return True
from learning.learner import Learner
from utils import error
import numpy as np
from bundle.datausages import *
from bundle.datatypes import *

class SupervisedLearner(Learner):
    """Class for learners with generic ground truth"""
    def __init__(self, consumes=[Numeric.name, GroundTruth.name]):
        Learner.__init__(self, consumes)

    def assign_current_run_data(self, iteration_index, trainval):
        """Sets data for the current run"""
        # let learner handle indexes to data
        super().assign_current_run_data(iteration_index, trainval)
        # get label containers and print statistics
        self.train_targets, self.val_targets, self.test_targets = self.validation.get_run_labels(iteration_index, trainval)
        # update indexes to label of the evaluator: the trainval indexes
        train_idx, val_idx = trainval
        test_label_index = np.arange(len(self.test_targets)) if not self.validation.use_for_testing else val_idx
        self.evaluator.update_reference_targets(train_idx, test_label_index)

    def get_component_inputs(self):
        """Component processing for ground truth data"""
        # get gt
        self.process_ground_truth_input()
        # get indexes
        super().get_component_inputs()


    def process_ground_truth_input(self):
        targets = self.data_pool.request_data(Text, GroundTruth, usage_matching="subset", client=self.name, must_be_single=(not self.model_loaded))
        if targets:
            self.targets = targets.data
            self.target_indices = targets.get_usage(Indices.name)
        else:
            self.targets = Numeric([])
            self.target_indices = Indices([])

    def attach_evaluator(self):
        super().attach_evaluator()
        train_target, test_target = self.get_train_test_targets()
        self.evaluator.set_targets(train_target, test_target)
        # self.evaluator.set_labelling(train_target, self.tokenizer.get_vocab().values(), do_multilabel=True)

    def check_sanity(self):
        super().check_sanity()
        # if we're to build the model, check that everything matches
        if not self.model_loaded:
            if len(self.train_embedding_index) != len(self.target_train_embedding_index):
                error(f"Supplied train data-gt instance number mismatch: {len(self.train_embedding_index)} data and {len(self.target_train_embedding_index)}")
            if len(self.test_embedding_index) != len(self.target_test_embedding_index):
                error(f"Supplied test data-gt instance number mismatch: {len(self.test_embedding_index)} data and {len(self.target_test_embedding_index)}")


    def get_train_test_targets():
        error("Attempted to access target getter from base class")
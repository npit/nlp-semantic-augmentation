from learning.learner import Learner
from utils import error
import numpy as np
from bundle.datausages import *
from bundle.datatypes import *

class SupervisedLearner(Learner):
    """Class for learners with generic ground truth"""
    targets_data = None

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

    def get_ground_truth(self):
        """GT retrieval"""
        return self.targets

    def process_ground_truth_input(self):
        # if we have to train the model, we can load more than one ground truth instance (i.e. train & test)
        need_instances = not self.model_loaded
        # fetch any type of ground 
        # targets = self.data_pool.request_data(None, GroundTruth.get_subclasses(), usage_matching="any", client=self.name, must_be_single=single_instances)
        targets = self.data_pool.request_data(None, GroundTruth.get_subclasses(), usage_matching="any", client=self.name, must_be_single=need_instances)
        self.targets_data = targets
        if targets:
            self.targets = targets.data
            self.target_indices = targets.get_usage(Indices)
        else:
            self.targets = Numeric([])
            self.target_indices = Indices([], [])

    # def attach_evaluator(self):
    #     super().attach_evaluator()
    #     train_target, test_target = self.get_train_test_targets()
    #     self.evaluator.set_targets(train_target, test_target)
    #     # self.evaluator.set_labelling(train_target, self.tokenizer.get_vocab().values(), do_multilabel=True)

    # def check_sanity(self):
    #     super().check_sanity()
    #     # if we're to build the mode

    def get_train_test_targets():
        error("Attempted to access target getter from base class")

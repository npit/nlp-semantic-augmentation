from bundle.bundle import DataPool
from bundle.datatypes import *
from bundle.datausages import *

from utils import error, warning
from evaluation.evaluator import Evaluator, get_random_predictions
import numpy as np
import rouge
from sklearn import metrics
import defs



class UnsupervisedEvaluator(Evaluator):
    """Evaluation class for unsupervised learning results"""
    name = "unsupervised_evaluator"

    consumes = Numeric.name
    available_measures = ("silhouette")

    def __init__(self, config):
        self.config = config
        self.measure_funcs = {"silhouette": UnsupervisedEvaluator.compute_silhouette}
        super().__init__(config)


    def get_component_inputs(self):
        """Get inputs for unsupervised evaluation"""

        super().get_component_inputs()
        # get learner inputs
        dp = self.data_pool.request_data(None, Indices, usage_matching="subset", client=self.name, usage_exclude=[GroundTruth, Predictions])

        self.indices = dp.get_usage(Indices).instances
        self.data = dp.data.instances

        # train_idx = self.indices.get_tag_instances(defs.roles.train)
        # test_idx = self.indices.get_tag_instances(defs.roles.test)
        # # fetch learner train, test and prediction data
        # test = data.get_instances(test_idx)
        # if test:
        #     self.data = test
        # else:
        #     self.data = data.get_instances(train_idx)
        if len(self.preds) > 1:
            self.num_eval_iterations = len(self.preds)

    # def evaluate_predictions(self, preds, out_dict):
    #     """Perform all evaluations on given predictions"""
    #     # iterate over all measures
    #     for measure in self.measures:
    #         out_dict[measure] = {self.iteration_name:[]}
    #         # iterate over data, corresponding to folds
    #         for i, _ in enumerate(self.preds):
    #             idx = self.indices[i]
    #             data = self.data[idx]
    #             preds = self.preds[i]
    #             score = self.evaluate_measure((data, preds), measure)
    #             out_dict[measure][self.iteration_name].append(score)
    #         # fold aggregations
    #         for name, func in zip("mean std var".split(), (np.mean, np.std, np.var)):
    #             out_dict[name] = func(out_dict[measure][self.iteration_name])
    def get_evaluation_input(self, prediction_index):
        """Retrieve input data to evaluation function(s)

        For unsupervised evaluation, retrieve input data and predictions
        """
        preds = super().get_evaluation_input(prediction_index)
        idx = self.indices[prediction_index]
        data = self.data[idx]
        return (data, preds)

    # individual eval. methods:
    ###########

    def compute_silhouette(self, data):
        """Compute the silhouette coefficient
        data: tuple of input_data, predictions
        input_data: n_instances x n_features matrix
        predictions: n_instances x n_cluster_distances matrix
        """
        data, preds = data
        preds = np.argmax(preds, axis=1)
        return metrics.silhouette_score(data, preds)


    def evaluate_measure(self, data, measure):
        """Dict wrapper"""
        score = super().evaluate_measure(data, measure)
        return {"score": score}
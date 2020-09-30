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

    @staticmethod
    def matches_config(config):
        """Determine whether the evaluator is applicable wrt the input config"""
        return (not config.measures) or all(me in UnsupervisedEvaluator.available_measures for me in config.measures)

    def __init__(self, config):
        self.config = config
        self.measure_funcs = {"silhouette": UnsupervisedEvaluator.compute_silhouette}

        self.measures = self.config.measures

        super().__init__(config)


    def get_component_inputs(self):
        """Get inputs for unsupervised evaluation"""

        # get learner inputs
        dp = self.data_pool.request_data(None, Indices, usage_matching="subset", client=self.name, usage_exclude=[GroundTruth, Predictions])
        preds = self.data_pool.request_data(None, Predictions, usage_matching="exact", client=self.name)

        self.indices = dp.get_usage(Indices.name).instances
        self.data = dp.data.instances

        # train_idx = self.indices.get_role_instances(defs.roles.train)
        # test_idx = self.indices.get_role_instances(defs.roles.test)
        # # fetch learner train, test and prediction data
        # test = data.get_instance(test_idx)
        # if test:
        #     self.data = test
        # else:
        #     self.data = data.get_instance(train_idx)
        self.preds = preds.data.instances
        if len(self.preds) > 1:
            self.num_eval_iterations = len(self.preds)

    def make_evaluations(self):
        """Do evaluations"""
        self.results = {"run":{}, "random": {}}
        self.evaluate_predictions(self.preds, self.results["run"])
        # baselines
        rand_preds = [get_random_predictions(self.preds[0].shape) for _ in range(self.num_eval_iterations)]
        self.evaluate_predictions(rand_preds, self.results["random"])

    def get_results(self):
        return self.results

    def evaluate_predictions(self, preds, out_dict):
        """Perform all evaluations on given predictions"""
        # iterate over all measures
        for measure in self.measures:
            out_dict[measure] = {self.iteration_name:[]}
            # iterate over data, corresponding to folds
            for i, _ in enumerate(self.preds):
                idx = self.indices[i]
                data = self.data[idx]
                preds = self.preds[i]
                score = self.evaluate_measure((data, preds), measure)
                out_dict[measure][self.iteration_name].append(score)
            # fold aggregations
            for name, func in zip("mean std var".split(), (np.mean, np.std, np.var)):
                out_dict[name] = func(out_dict[measure][self.iteration_name])

    def evaluate_measure(self, data, measure):
        """Apply an evaluation measure"""
        try:
            return self.measure_funcs[measure](self, data)
        except KeyError:
            warning(f"Unavailable measure: {measure}")
            return -1.0

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


from evaluation.unsupervised_evaluator import UnsupervisedEvaluator
from evaluation.supervised_evaluator import SupervisedEvaluator
from utils import error

class Instantiator:
    component_name = "evaluator"

    @staticmethod
    def create(config):
        """Function to instantiate an evaluator"""
        candidates = [UnsupervisedEvaluator, SupervisedEvaluator]

        # instantiate non-neural candidates
        for candidate in candidates:
            if candidate.matches_config(config):
                return candidate(config)
        error(f"Inconsistent evaluator with config {config.__dict__}")

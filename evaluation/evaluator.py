from component.component import Component
from serializable import Serializable
from bundle.datatypes import *
from bundle.datausages import *
import numpy as np
from utils import write_pickled, error
from os.path import join

def get_random_predictions(shp):
    """Get random predictions wrt. the input shape"""
    return np.random.rand(*shp)

class Evaluator(Serializable):
    """Generic evaluation class"""
    component_name = "evaluator"

    # predictions storage
    predictions = None
    predictions_instance_indexes = None
    confusion_matrices = None

    num_eval_iterations = 1
    iteration_name = "folds"


    def __init__(self, config):
        self.config = config
        self.output_folder = join(config.folders.run, "results")

    def produce_outputs(self):
        self.make_evaluations()

    def save_outputs(self):
        """Write the evaluation results"""
        write_pickled(join(self.output_folder, "results.pkl"), self.get_results())

    def set_component_outputs(self):
        pass

    def get_component_inputs(self):
        error("Attempted to get inputs via abstract evaluator.")

    def attempt_load_model_from_disk(self, force_reload=False, failure_is_fatal=False):
        # building an evaluator model is not defined -- failure never fatal
        return super().attempt_load_model_from_disk(force_reload=force_reload, failure_is_fatal=False)

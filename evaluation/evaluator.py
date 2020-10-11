from component.component import Component
from serializable import Serializable
from bundle.datatypes import *
from bundle.datausages import *
import numpy as np
from utils import write_pickled, error, warning, info
from os.path import join
import json

import pandas as pd

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
    iteration_aggregations = ("mean", "std", "var")
    iteration_aggregation_funcs = (np.mean, np.std, np.var)
    iterations_alias = "folds" # alias for iterations

    print_precision = 3

    @classmethod
    def matches_config(cls, config):
        """Determine whether the evaluator is applicable wrt the input config"""
        return (not config.measures) or all(me in cls.available_measures for me in config.measures)

    def __init__(self, config):
        self.config = config
        self.output_folder = join(config.folders.run, "results")

        self.print_measures = self.config.measures
        self.print_aggregations = self.config.iter_aggregations
        if self.print_aggregations is None:
            self.print_aggregations = ("mean", "std")

    def produce_outputs(self):
        self.results = {"run":{}, "random": {}}
        prediction_shape = self.predictions[0].shape
        self.predictions = self.preprocess_predictions(self.predictions)

        info("Evaluating run")
        self.evaluate_predictions(self.predictions, "run")
        # baselines
        info("Evaluating random baseline")
        rand_preds = [get_random_predictions(prediction_shape) for _ in range(self.num_eval_iterations)]
        rand_preds = self.preprocess_predictions(rand_preds)

        self.evaluate_predictions(rand_preds, "random")
        # info(json.dumps(self.results, indent=2))
        info("Displaying results")
        self.print_results()

    def print_results(self):
        for results_type in self.results:
            # info(f"{results_type} | {self.iterations_alias} {' '.join(self.iteration_aggregations)}:")
            df = pd.DataFrame(columns=[self.iterations_alias] + list(self.iteration_aggregations))
            for measure in self.results[results_type]:
                self.print_measure(measure, self.results[results_type][measure], df=df)

            # df.style.format('{:.4f}')
            pd.options.display.max_rows = 999
            pd.options.display.max_columns = 999
            pd.options.display.precision = 3

            df = self.set_printable_info(df)
            info(f"{results_type}:")
            info("-------------------------------")
            for k in df.to_string().split("\n"):
                info(k)
            info("-------------------------------")

    def compute_additional_info(self):
        """Print additional information on the run"""
        pass

    def set_printable_info(self, df):
        # iteration aggregations
        df.index.name = "measure"
        df = df.drop(columns=[x for x in df.columns if x not in self.print_aggregations])
        # measures
        df = df.drop(index=[x for x in df.index if not any (x.startswith(k) for k in self.print_measures)])
        return df

    def print_measure(self, measure, ddict, print_override=None, df=None):
        """Print measure results, aggregating over prediction iterations"""
        aggrs = list(ddict.keys())
        values = []

        if print_override is not None:
            prefix = print_override
        else:
            prefix = measure

        for iteration_aggr in aggrs:
            values.append(ddict[iteration_aggr])
        # info(f"{prefix} {values}")
        df.loc[prefix] = [self.round(x) for x in values]

    def round(self, val):
        if type(val) is list:
            return [self.round(x) for x in val]
        return np.round(val, self.print_precision)

    def evaluate_predictions(self, preds, key):
        """Perform all evaluations on given predictions"""
        out_dict = self.results[key]
        # iterate over all measures
        for measure in self.available_measures:
            result = self.evaluate_measure(preds,measure)
            out_dict[measure] = result
        self.compute_additional_info(preds, key)

    def aggregate_iterations(self, ddict):
        """Aggregate multi-value entries"""
        values = ddict[self.iterations_alias]
        # iterations aggregations
        for name, func in zip(self.iteration_aggregations, self.iteration_aggregation_funcs):
            ddict[name] = func(values, axis=0)
        return ddict

    def evaluate_measure(self, predictions, measure):
        """Apply an evaluation measure"""
        try:
            data = self.get_evaluation_input(predictions)
            return self.measure_funcs[measure](data)
        except KeyError:
            warning(f"Unavailable measure: {measure}")
            return -1.0

    def save_model(self):
        """Undefined model IO"""
        pass
    def load_model(self):
        """Undefined model IO"""
        return False

    def save_outputs(self):
        """Write the evaluation results"""
        write_pickled(join(self.output_folder, "results.pkl"), self.get_results())

    def get_evaluation_input(self, prediction_index, predictions=None):
        """Fetch predictions"""
        if predictions is None:
            predictions = self.predictions
        # get a batch of predictions
        return predictions[prediction_index]

    def set_component_outputs(self):
        pass

    def get_component_inputs(self):
        # fetch predictions
        preds = self.data_pool.request_data(None, Predictions, usage_matching="exact", client=self.name)
        self.predictions = preds.data.instances
        self.reference_indexes = preds.get_usage(Predictions).instances

    def get_results(self):
        return self.results


    def attempt_load_model_from_disk(self, force_reload=False, failure_is_fatal=False):
        # building an evaluator model is not defined -- failure never fatal
        return super().attempt_load_model_from_disk(force_reload=force_reload, failure_is_fatal=False)

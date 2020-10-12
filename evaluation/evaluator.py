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
        self.results = {"run":{}}

        self.evaluate_baselines()

        info("Evaluating run")
        self.predictions = self.preprocess_predictions(self.predictions)
        self.evaluate_predictions(self.predictions, "run")



        # info(json.dumps(self.results, indent=2))
        info("Displaying results")
        self.print_results()

    def evaluate_baselines(self):
        # baselines
        prediction_shape = self.predictions[0].shape

        info("Evaluating random baseline")
        self.results["random"] = {}
        rand_preds = [get_random_predictions(prediction_shape) for _ in range(self.num_eval_iterations)]
        rand_preds = self.preprocess_predictions(rand_preds)
        self.evaluate_predictions(rand_preds, "random", override_tags_roles=("model", "rand-baseline"))


    def print_results(self):
        for results_type in self.results:
            # info(f"{results_type} | {self.iterations_alias} {' '.join(self.iteration_aggregations)}:")
            for tag in self.results[results_type]:
                for role in self.results[results_type][tag]:
                    df = pd.DataFrame(columns=[self.iterations_alias] + list(self.iteration_aggregations))
                    for measure in self.results[results_type][tag][role]:
                        self.print_measure(measure, self.results[results_type][tag][role][measure], df=df)

                    # display the result
                    pd.options.display.max_rows = 999
                    pd.options.display.max_columns = 999
                    pd.options.display.precision = 3

                    df = self.set_printable_info(df)
                    if df.size == 0:
                        continue
                    info(f"{results_type}-{tag}-{role}:")
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

    def evaluate_predictions(self, predictions, key, override_tags_roles=None):
        """Perform all evaluations on given predictions"""
        out_dict = self.results[key]
        if override_tags_roles is None:
            tags, roles = self.tags, self.roles
        else:
            tags, roles = override_tags_roles
            tags, roles = [tags], [roles]

        # group by tag
        for tag in set(tags):
            tag_idx = [i for i, t in enumerate(tags) if t == tag]
            out_dict[tag] = {}
            # group by role
            for role in set(roles):
                idx = [i for i in tag_idx if roles[i] == role]
                if not idx:
                    continue
                out_dict[tag][role] = {}

                # iterate over all measures
                for measure in self.available_measures:
                    result = self.evaluate_measure(predictions, idx, measure)
                    out_dict[tag][role][measure] = result
                self.compute_additional_info(predictions, idx, f"{key}-{tag}-{role}")

        if len(tags) > 1:
            out_dict["all_tags"] = {}
            self.aggregate_tags(tags, roles, out_dict)

    def aggregate_tags(self, tags, roles, out_dict):
        # perform an aggregation across all tags as well, if applicable
        for role in set(roles):
            out_dict["all_tags"][role] = {}
            for measure in self.available_measures:
                tag_values = [out_dict[t][role][measure][self.iterations_alias] for t in tags]
                # flatten
                tag_values = [x for v in tag_values for x in v]
                out_dict["all_tags"][role][measure] = {self.iteration_alias: tag_values}
                self.aggregate_iterations(out_dict["all_tags"][role][measure])

    def aggregate_iterations(self, ddict):
        """Aggregate multi-value entries"""
        values = ddict[self.iterations_alias]
        # iterations aggregations
        for name, func in zip(self.iteration_aggregations, self.iteration_aggregation_funcs):
            ddict[name] = func(values, axis=0)
        return ddict

    def evaluate_measure(self, predictions, index, measure):
        """Apply an evaluation measure"""
        try:
            data = self.get_evaluation_input(predictions, index)
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
        # get a batch of predictions according to the input index
        return predictions[prediction_index]

    def set_component_outputs(self):
        pass

    def get_component_inputs(self):
        # fetch predictions
        preds_datapack = self.data_pool.request_data(None, Predictions, usage_matching="exact", client=self.name)
        self.predictions = preds_datapack.data.instances
        self.reference_indexes = preds_datapack.get_usage(Predictions).instances

        preds_usage = preds_datapack.get_usage(Predictions)
        self.roles, self.tags = preds_usage.roles, preds_usage.tags

    def get_results(self):
        return self.results


    def attempt_load_model_from_disk(self, force_reload=False, failure_is_fatal=False):
        # building an evaluator model is not defined -- failure never fatal
        return super().attempt_load_model_from_disk(force_reload=force_reload, failure_is_fatal=False)

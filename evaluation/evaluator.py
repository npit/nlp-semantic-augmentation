from component.component import Component
from collections import defaultdict
from serializable import Serializable
from bundle.datatypes import *
from bundle.datausages import Predictions
import numpy as np
from utils import write_pickled, error, warning, info
from os.path import join
import json
import defs

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
        Serializable.__init__(self, "")
        self.output_folder = join(config.folders.run, "results")

        self.print_measures = self.config.measures
        self.print_aggregations = self.config.iter_aggregations
        if self.print_aggregations is None:
            self.print_aggregations = ("mean", "std")

    def is_baseline_run(self, run_type):
        return run_type.startswith("random")

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
        prediction_shape = self.predictions.shape

        info("Evaluating random baseline")
        self.results["random"] = {}
        rand_preds = [get_random_predictions(prediction_shape) for _ in range(self.num_eval_iterations)][0]
        rand_preds = self.preprocess_predictions(rand_preds)
        self.evaluate_predictions(rand_preds, "random", override_tags_roles=("model", "rand-baseline"))

    def print_results(self):
        # print results per runtype / tag / role
        for results_type in self.results:
            num_tags = len(self.results[results_type].keys())
            for tag in self.results[results_type]:
                for role in self.results[results_type][tag]:
                    if tag == "all_tags":
                        # include aggregation stats
                        df = pd.DataFrame(columns=[self.iterations_alias] + list(self.iteration_aggregations))
                    else:
                        df = pd.DataFrame(columns=["score"])
                    for measure in self.results[results_type][tag][role]:
                        self.print_measure(measure, self.results[results_type][tag][role][measure], df=df)

                    if num_tags > 1:
                        if not self.should_print_this(results_type, tag):
                            continue
                    pref = f"{results_type}-{tag}-{role}:"
                    if self.is_baseline_run(results_type):
                        pref = "(baseline) " + pref
                    self.print_results_dataframe(df, pref)

    def print_results_dataframe(self, df, prefix):
        """Display results"""
        # display the result
        pd.options.display.max_rows = 999
        pd.options.display.max_columns = 999
        pd.options.display.precision = 3

        df = self.set_printable_info(df)
        if df.size == 0:
            return
        info(f"{prefix}")
        info("-------------------------------")
        for k in df.to_string().split("\n"):
            info(k)
        info("-------------------------------")

    def compute_additional_info(self, preds, idx, prefix, do_print=True):
        """Print additional information on the run"""
        pass

    def get_print_dataframe_index_name(self):
        return "measure"

    def set_printable_info(self, df):
        # iteration aggregations
        df.index.name = self.get_print_dataframe_index_name()
        df = df.drop(columns=[x for x in df.columns if x not in self.print_aggregations and x != "score"])
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

    def evaluate_predictions(self, input_predictions, run_type, override_tags_roles=None):
        """Perform all evaluations on given predictions
        Perform a two-step hierarchical aggregation:
        Outer: any tag that's not inner (e.g. different models / folds)
        Inner: "train", "test"
        """
        out_dict = self.results[run_type]
        # outer evaluation tags: model instances
        outer = [t for t in self.tags if t.startswith("model") and not t.endswith(defs.roles.inputs)]
        inner = [t for t in self.tags if t in [defs.roles.train, defs.roles.val, defs.roles.test]]

        # collect indexes to across outer tags to produce, e.g. total <train> performance
        total_idxs_inner = defaultdict(list)
        has_multiple_models = len(outer) > 1

        for outer_tag in outer:
            out_idx = self.indexes[self.tags.index(outer_tag)]
            out_dict[outer_tag] = {}

            for inner_tag in inner:
                # get prediction indexes
                in_idx = self.indexes[self.tags.index(inner_tag)]
                joint_idx = np.intersect1d(out_idx, in_idx)
                current_predictions = input_predictions[joint_idx]
                if len(current_predictions) == 0:
                    continue

                total_idxs_inner[inner_tag].append(joint_idx)

                out_dict[outer_tag][inner_tag] = {}

                for measure in self.available_measures:
                    result = self.evaluate_measure(current_predictions, joint_idx, measure, tag_info=(outer_tag, inner_tag))
                    out_dict[outer_tag][inner_tag][measure] = result
                do_print = (not has_multiple_models) or self.should_print_this(run_type, outer_tag)
                self.compute_additional_info(current_predictions, joint_idx, f"{run_type}-{outer_tag}-{inner_tag}", do_print=do_print)

        if has_multiple_models:
            # aggregate
            out_dict["all_tags"] = {}
            self.aggregate_tags(outer, inner, out_dict)
            for o in total_idxs_inner:
                self.compute_additional_info(input_predictions, np.concatenate(total_idxs_inner[o]), f"{run_type}-{o}-all_tags", do_print=not self.is_baseline_run(run_type))
        print()

    def should_print_this(self, run_type, tag):
        # whether or not to print individual models (e.g. individual folds)
        if tag != "all_tags":
            # skip if it is set as such or we are at a baseline
            if (not self.config.print_individual_models) or self.is_baseline_run(run_type):
                return False
        return True



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
        preds_datapack = self.data_pool.request_data(None, Predictions, usage_matching="subset", client=self.name)
        # get numeric prediction data
        self.predictions = preds_datapack.data.instances

        # get indices
        preds_usage = preds_datapack.get_usage(Predictions)
        self.tags = preds_usage.tags
        # ensure unique tags
        error(f"Evaluation requires unique tagset, got {self.tags}", len(self.tags) != len(set(self.tags)))
        self.indexes = preds_usage.instances


    def get_results(self):
        return self.results


    def attempt_load_model_from_disk(self, force_reload=False, failure_is_fatal=False):
        # building an evaluator model is not defined -- failure never fatal
        return super().attempt_load_model_from_disk(force_reload=force_reload, failure_is_fatal=False)

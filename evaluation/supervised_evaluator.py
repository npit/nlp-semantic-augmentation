from bundle.bundle import DataPool
from evaluation.evaluator import Evaluator
# import rouge
from bundle.datatypes import *
from bundle.datausages import *
from collections import defaultdict

import numpy as np
from sklearn import metrics
from utils import info, count_occurences
from sklearn.dummy import DummyClassifier

class SupervisedEvaluator(Evaluator):
    """Evaluator for supervised tasks"""

    name = "supervised_evaluator"
    consumes = [Numeric.name, GroundTruth.name]
    available_measures = ("rouge", "f1", "accuracy", "precision", "recall")

    labels_info = None
    num_max_print_labels = 10
    available_label_aggregations = ["micro", "macro", "weighted", "none"]

    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.measure_funcs = {"rouge": self.compute_rouge,
                              "f1": self.compute_f1,
                              "accuracy": self.compute_accuracy,
                              "precision": self.compute_precision,
                              "recall": self.compute_recall
        }
        self.print_label_aggregations = self.config.label_aggregations
        if self.print_measures is None:
            self.print_measures = ("f1", "accuracy")
        if self.print_label_aggregations is None:
            self.print_label_aggregations = ("micro", "macro")
        undef_measures = [x for x in self.print_measures if x not in self.available_measures]
        error(f"Undefined specified measure(s): {undef_measures}", undef_measures)
        undef_lbl_aggr = [x for x in self.print_label_aggregations if x not in self.available_label_aggregations]
        error(f"Undefined specified label aggregation(s): {undef_lbl_aggr}", undef_lbl_aggr)
        # dedicated container for majority baseline (that's index-dependent)
        self.results_majority_baseline = {}


    def get_component_inputs(self):
        """Get inputs for unsupervised evaluation"""
        # anything but ground truth
        super().get_component_inputs()
        matches = self.data_pool.request_data(None, [Labels, Indices], usage_matching="exact", client=self.name, on_error_message="Failed to find ground truth labels.")
        self.labels = matches.data
        self.labels_info = matches.get_usage(Labels)
        # TODO subclass
        # perform single-label transformations
        self.labels.instances = np.concatenate(self.labels.instances)

    def set_printable_info(self, df):
        df = super().set_printable_info(df)
        # label aggregations
        df = df.drop(index=[x for x in df.index if not any (x.endswith(k) for k in self.print_label_aggregations)])
        # df.index.name = "measure_label-aggr"
        return df

    def get_evaluation_input(self, predictions, indexes):
        """Retrieve input data to evaluation function(s)

        For supervised evaluation, retrieve ground truth and predictions
        """
        # # fetch predictions instance
        # preds = super().get_evaluation_input(prediction_index, predictions)
        # # fetch the indices to the labels the prediction batch corresponds to
        # idx = self.reference_indexes[prediction_index]
        # compute the argmax
        # # fetch the labels wrt. the indices
        labels = self.labels.get_slice(indexes)

        # preds = self.preprocess_predictions(preds)
        # labels = self.preprocess_ground_truth(labels)
        return (labels, predictions)


    # measure functions

    def preprocess_predictions(self, predictions):
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def preprocess_ground_truth(self, gt):
        gt = np.concatenate(gt)
        return gt

    computed_maj_baseline_for_indexes = set()

    def evaluate_measure(self, predictions, indexes, measure, tag_info=None):
        """Evaluate a measure on input data over label aggregations

        Arguments:
        Returns:
        res (dict): Dictionary like {"label_aggr1": <score>, "label_aggr2": <score>}
        """
        res = {}
        for lbl_aggr in self.available_label_aggregations:
            key = lbl_aggr
            res[key] = {self.iterations_alias:[]}
            gt, preds = self.get_evaluation_input(predictions, indexes)
            score = self.measure_funcs[measure](gt, preds, lbl_aggr if lbl_aggr != "none" else None)
            res[key][self.iterations_alias].append(score)
            # # compute aggregation of the multiple-iterations
            # res[lbl_aggr] = self.aggregate_iterations(res[lbl_aggr])

        # by the way -- if we need to evaluate the majority baseline,
        # since it's dependent on the input labels portion
        # do it sneakily here instead.
        self.compute_majority_baseline(measure, predictions, indexes, tag_info)
        return res


    def compute_majority_baseline(self, measure, predictions, indexes, tag_info):
        """
        Compute a majority-based baseline wrt. input indexes. Store results in
        a dict specified by the tag information
        """
        # keep track of baseline'd index collection
        idxs_hash = hash(str(indexes) + str(indexes) + str(tag_info) + str(measure))
        if idxs_hash not in self.computed_maj_baseline_for_indexes:
            self.computed_maj_baseline_for_indexes.add(idxs_hash)
            # get gt for that chunk
            gt, _ = self.get_evaluation_input(predictions, indexes)
            # make dummy classifier
            dc = DummyClassifier(strategy="stratified")
            dc.fit(gt, y=gt)
            maj_preds = dc.predict(predictions)
            # evaluate the maj predictions
            res = self.evaluate_measure(maj_preds, indexes, measure, tag_info)
            curr_dict = self.results_majority_baseline
            for t in tag_info:
                if t not in curr_dict:
                    curr_dict[t] = {}
                curr_dict = curr_dict[t]
            curr_dict[measure] = res

    def evaluate_baselines(self):
        # the majority baseline is computed inline each evaluate_measure function call,
        # for each separate input indexes (corresponding to different ground truth distros)
        super().evaluate_baselines()
        # do aggregation
        if len(self.results_majority_baseline) > 1:
            outer = list(self.results_majority_baseline.keys())
            inner = list(self.results_majority_baseline[outer[0]].keys())
            self.results_majority_baseline["all_tags"] = {}
            self.aggregate_tags(outer, inner, self.results_majority_baseline)

        # add it to the container
        self.results["majority"] = self.results_majority_baseline

    def is_baseline_run(self, run_type):
        return super().is_baseline_run(run_type) or run_type.startswith("majority")

    def aggregate_tags(self, tags, roles, out_dict):
        # perform an aggregation across all tags as well, if applicable
        for role in set(roles):
            out_dict["all_tags"][role] = {}
            for measure in self.available_measures:
                out_dict["all_tags"][role][measure] = {}
                for laggr in self.available_label_aggregations:
                    out_dict["all_tags"][role][measure][laggr] = {}
                    tag_values = [out_dict[t][role][measure][laggr][self.iterations_alias] for t in set(tags)]
                    # flatten
                    tag_values = [x for v in tag_values for x in v]
                    out_dict["all_tags"][role][measure][laggr] = {self.iterations_alias: tag_values}
                    self.aggregate_iterations(out_dict["all_tags"][role][measure][laggr])

    def compute_additional_info(self, predictions, indexes, key, do_print=True):
        # compute label distributions
        # tuplelist to string
        tl2s = lambda tlist: ", ".join(f"({x} ({self.labels_info.label_names[x]}): {y})" for (x,y) in tlist[:self.num_max_print_labels])

        gt, preds = self.get_evaluation_input(predictions, indexes)
        if do_print:
            info(f"{key} | predictions ({len(preds)} instances) Top-{self.num_max_print_labels} label distros (index/labelname):count :")
        gt_distr, preds_distr = count_occurences(gt), count_occurences(preds)

        if do_print:
            info(f"gt:    {tl2s(gt_distr)}")
            info(f"preds: {tl2s(preds_distr)}")


    def get_print_dataframe_index_name(self):
        return "measure-labelaggregation"

    def print_measure(self, measure, ddict, df=None):
        """Print measure results, aggregating over prediction iterations"""
        for lbl_agg in self.print_label_aggregations:
            super().print_measure(measure, ddict[lbl_agg], print_override=f"{measure}-{lbl_agg}", df=df)

    def compute_f1(self, gt, preds, lbl_aggr):
        return metrics.f1_score(gt, preds, average=lbl_aggr)

    def compute_precision(self, gt, preds, lbl_aggr):
        return metrics.precision_score(gt, preds, average=lbl_aggr)

    def compute_recall(self, gt, preds, lbl_aggr):
        return metrics.recall_score(gt, preds, average=lbl_aggr)

    def compute_accuracy(self, gt, preds, lbl_aggr):
        return metrics.accuracy_score(gt, preds)

    def compute_rouge(self, gt, preds, lbl_aggr):
        """Compute rouge"""
        return -1
        # gt, preds = data
        # preds = np.argmax(preds, axis=1)
        # maxlen = preds.shape[-1]
        # evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True, length_limit=maxlen,
        #         length_limit_type='words',
        #         apply_avg=True,
        #         alpha=0.5, # Default F1_score
        #         weight_factor=1.2, stemming=True)
        # scores = evaluator.get_scores(gt, preds)
        # return scores


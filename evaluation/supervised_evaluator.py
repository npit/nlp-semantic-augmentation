from bundle.bundle import DataPool
from evaluation.evaluator import Evaluator
import rouge
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
    label_aggregations = ["micro", "macro", "weighted", "none"]

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


    def get_component_inputs(self):
        """Get inputs for unsupervised evaluation"""
        # anything but ground truth
        super().get_component_inputs()
        matches = self.data_pool.request_data(None, Labels, usage_matching="subset", client=self.name)
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

    def get_evaluation_input(self, prediction_index, predictions=None):
        """Retrieve input data to evaluation function(s)

        For supervised evaluation, retrieve ground truth and predictions
        """
        if predictions is None:
            predictions = self.predictions
        # fetch predictions instance
        preds = super().get_evaluation_input(prediction_index, predictions)
        # fetch the indices to the labels the prediction batch corresponds to
        idx = self.reference_indexes[prediction_index]
        # fetch the labels wrt. the indices
        labels = self.labels.get_slice(idx)

        # preds = self.preprocess_predictions(preds)
        # labels = self.preprocess_ground_truth(labels)
        return (labels, preds)


    # measure functions

    def preprocess_predictions(self, predictions):
        for i in range(len(predictions)):
            preds = predictions[i]
            if preds.ndim == 1:
                preds = np.expand_dims(preds, axis=0)
            predictions[i] = np.argmax(preds, axis=1)
        return predictions

    def preprocess_ground_truth(self, gt):
        gt = np.concatenate(gt)
        return gt

    def evaluate_measure(self, predictions, indexes, measure):
        """Evaluate a measure on input data over label aggregations

        Arguments:
        Returns:
        res (dict): Dictionary like {"label_aggr1": <score>, "label_aggr2": <score>}
        """
        res = {}
        for lbl_aggr in self.label_aggregations:
            res[lbl_aggr] = {self.iterations_alias:[]}
            iter_values = res[lbl_aggr][self.iterations_alias]

            for i in indexes:
                gt, preds = self.get_evaluation_input(i, predictions=predictions)
                score = self.measure_funcs[measure](gt, preds, lbl_aggr if lbl_aggr is not "none" else None)
                iter_values.append(score)
            # compute aggregation of the multiple-iterations
            res[lbl_aggr] = self.aggregate_iterations(res[lbl_aggr])
        return res


    def aggregate_tags(self, tags, roles, out_dict):
        # perform an aggregation across all tags as well, if applicable
        for role in set(roles):
            out_dict["all_tags"][role] = {}
            for measure in self.available_measures:
                out_dict["all_tags"][role][measure] = {}
                for laggr in self.label_aggregations:
                    out_dict["all_tags"][role][measure][laggr] = {}
                    tag_values = [out_dict[t][role][measure][laggr][self.iterations_alias] for t in set(tags)]
                    # flatten
                    tag_values = [x for v in tag_values for x in v]
                    out_dict["all_tags"][role][measure][laggr] = {self.iterations_alias: tag_values}
                    self.aggregate_iterations(out_dict["all_tags"][role][measure][laggr])


    def compute_additional_info(self, predictions, index, key):
        # compute label distributions
        # tuplelist to string
        tl2s = lambda tlist: ", ".join(f"({x}: {y})" for (x,y) in tlist)

        for i in index:
            gt, preds = self.get_evaluation_input(i, predictions=predictions)
            info(f"{key} | predictions batch #{i+1} ({len(preds)} instances) Label distros:")
            gt_distr, preds_distr = count_occurences(gt), count_occurences(preds)

            info(f"gt:    {tl2s(gt_distr)}")
            info(f"preds: {tl2s(preds_distr)}")

    def evaluate_baselines(self):
        super().evaluate_baselines()

        info("Evaluating majority baseline")
        self.results["majority"] = {}
        maj_preds = []
        for i in range(len(self.predictions)):
            _, gt = self.get_evaluation_input(i)
            dc = DummyClassifier(strategy="prior")
            dc.fit(gt, y=gt)
            maj_preds.append(dc.predict(self.predictions[i]))
        maj_preds = self.preprocess_predictions(maj_preds)
        self.evaluate_predictions(maj_preds, "majority", override_tags_roles=("model", "maj-baseline"))


    def print_measure(self, measure, ddict, df=None):
        """Print measure results, aggregating over prediction iterations"""
        for lbl_agg in self.label_aggregations:
            super().print_measure(measure, ddict[lbl_agg], print_override=f"{measure} {lbl_agg}", df=df)

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
        gt, preds = data
        preds = np.argmax(preds, axis=1)
        maxlen = preds.shape[-1]
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True, length_limit=maxlen, 
                length_limit_type='words',
                apply_avg=True,
                alpha=0.5, # Default F1_score
                weight_factor=1.2, stemming=True)
        scores = evaluator.get_scores(gt, preds)
        return scores

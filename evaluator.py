from utils import info, error, warning
import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from os.path import join
from utils import info, warning, error, one_hot, get_majority_label


class Evaluator:
    """Class to produce evaluation measures
    """

    # performance containers
    performance = {}
    cw_performance = {}

    # available measures, aggregations and run types
    measures = ["precision", "recall", "f1-score", "accuracy"]
    stats = ["mean", "var", "std", "folds"]
    run_types = ["random", "majority", "run"]
    multilabel_measures = ["ap", "roc_auc"]
    classwise_aggregations = ["macro", "micro", "classwise", "weighted"]

    # print opts
    print_precision = "{:.03f}"

    # constructor
    def __init__(self, config):
        """Evaluator constructor method"""
        self.config = config
        self.configure_evaluation_measures()

    def set_labels(self, test_labels, num_labels):
        self.test_labels = test_labels
        self.num_labels = num_labels

    def check_sanity(self, do_multilabel):
        """Evaluator sanity checking function"""
        # default measures if not preferred
        self.do_multilabel = do_multilabel
        if not self.preferred_measures:
            self.preferred_measures = self.measures if not do_multilabel else self.multilabel_measures
        else:
            # restrict as per labelling and sanity checks
            matching_measures = set(self.preferred_measures).intersection(self.measures) if not do_multilabel \
                else set(self.preferred_measures).intersection(self.multilabel_measures)
            if not matching_measures:
                error("Invalid preferred measures: {} for {} setting.".format(
                    self.preferred_measures,
                    "multilabel" if do_multilabel else "single-label"))
            self.preferred_measures = matching_measures

    # initialize evaluation containers and preferred evaluation printage
    def configure_evaluation_measures(self):
        """Method to initialize evaluation measure containers"""
        info("Creating learner: {}".format(self.config.learner.to_str()))
        for run_type in self.run_types:
            self.performance[run_type] = {}
            for measure in self.measures:
                self.performance[run_type][measure] = {}
                for aggr in self.classwise_aggregations:
                    self.performance[run_type][measure][aggr] = {}
                    for stat in self.stats:
                        self.performance[run_type][measure][aggr][stat] = None
                    self.performance[run_type][measure][aggr]["folds"] = []

            for measure in self.multilabel_measures:
                self.performance[run_type][measure] = {}
                self.performance[run_type][measure]["folds"] = []

            # remove undefined combos
            for aggr in [x for x in self.classwise_aggregations if x not in ["macro", "classwise"]]:
                del self.performance[run_type]["accuracy"][aggr]

        # print only these, from config
        self.preferred_types = self.config.print.run_types if self.config.print.run_types else self.run_types
        self.preferred_measures = self.config.print.measures if self.config.print.measures else []
        self.preferred_aggregations = self.config.print.aggregations if self.config.print.aggregations else self.classwise_aggregations
        self.preferred_stats = self.config.print.stats if self.config.print.stats else self.stats

        # sanity
        undefined = [x for x in self.preferred_types if x not in self.run_types]
        if undefined:
            error("undefined run type(s) in: {}, availables are: {}".format(undefined, self.run_types))
        undefined = [x for x in self.preferred_measures if x not in self.measures + self.multilabel_measures]
        if undefined:
            error("Undefined measure(s) in: {}, availables are: {}".format(undefined, self.measures + self.multilabel_measures))
        undefined = [x for x in self.preferred_aggregations if x not in self.classwise_aggregations]
        if undefined:
            error("Undefined aggregation(s) in: {}, availables are: {}".format(undefined, self.classwise_aggregations))

    # aggregated evaluation measure function shortcuts
    def get_pre_rec_f1(self, preds, metric, num_labels, gt=None):
        """Function to compute precision, recall and F1-score"""
        if gt is None:
            gt = self.test_labels
        cr = pd.DataFrame.from_dict(metrics.classification_report(gt, preds, output_dict=True))
        # get classwise, micro, macro, weighted
        keys = cr.keys()
        if len(keys) != num_labels + 3:
            existing_classes = [int(x) for x in keys[:-3]]
            warning("No predicted samples for classes: {}".format([x for x in range(num_labels) if x not in existing_classes]))
            existing_scores = cr.loc[metric].iloc[:-3].as_matrix()
            cw = np.zeros(num_labels, np.float32)
            for score_idx, class_number in enumerate(existing_classes):
                cw[class_number] = existing_scores[score_idx]
        else:
            cw = cr.loc[metric].iloc[:num_labels].as_matrix()
        mi = cr.loc[metric].iloc[-3]
        ma = cr.loc[metric].iloc[-2]
        we = cr.loc[metric].iloc[-1]
        return cw, mi, ma, we

    # get average accuracy
    def compute_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        return metrics.accuracy_score(gt, preds)

    # get class-wise accuracies
    def compute_classwise_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        cm = metrics.confusion_matrix(gt, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm.diagonal()

    # print performance across folds and compute foldwise aggregations
    def report_results(self, folds, write_folder):
        """Function to report learner results
        """
        info("==============================")
        info("{} performance {} across all [{}] folds:".format("/".join(self.preferred_types), "/".join(self.preferred_stats), folds))
        for run_type in self.run_types:
            if not self.do_multilabel:
                for measure in self.measures:
                    for aggr in self.classwise_aggregations:
                        if aggr not in self.performance[run_type][measure] or aggr == "classwise":
                            continue
                        container = self.performance[run_type][measure][aggr]
                        if not container:
                            continue
                        # add fold-aggregating performance information
                        self.performance[run_type][measure][aggr]["mean"] = np.mean(container["folds"])
                        self.performance[run_type][measure][aggr]["var"] = np.var(container["folds"])
                        self.performance[run_type][measure][aggr]["std"] = np.std(container["folds"])

                        # print the combination, if it's in the prefered stuff to print
                        if all([run_type in self.preferred_types, measure in self.preferred_measures, aggr in self.preferred_aggregations]):
                            scores_str = self.get_score_stats(container)
                            info("{:10} {:10} {:10} : {}".format(run_type, aggr, measure, scores_str))
            else:
                for measure in self.multilabel_measures:
                    container = self.performance[run_type][measure]
                    if not container:
                        continue
                    # add fold-aggregating performance information
                    self.performance[run_type][measure]["mean"] = np.mean(container["folds"])
                    self.performance[run_type][measure]["var"] = np.var(container["folds"])
                    self.performance[run_type][measure]["std"] = np.std(container["folds"])
                    # print, if it's prefered
                    if all([run_type in self.preferred_types, measure in self.preferred_measures]):
                        scores_str = self.get_score_stats(container)
                        info("{:10} {:10} : {}".format(run_type, measure, scores_str))

        if write_folder is not None:
            # write the results in csv in the results directory
            # entries in a run_type - measure configuration list are the foldwise scores, followed by the mean
            df = pd.DataFrame.from_dict(self.performance)
            df.to_csv(join(write_folder, "results.txt"))
            with open(join(write_folder, "results.pickle"), "wb") as f:
                pickle.dump(df, f)

    # print performance of the latest run
    def print_performance(self, current_run_descr, fold_index=0):
        info("---------------")
        info("Test results for {}:".format(current_run_descr))
        for rtype in self.preferred_types:
            if not self.do_multilabel:
                for measure in self.preferred_measures:
                    for aggr in self.preferred_aggregations:
                        # don't print classwise results or unedfined aggregations
                        if aggr not in self.performance[rtype][measure] or aggr == "classwise":
                            continue
                        container = self.performance[rtype][measure][aggr]
                        if not container:
                            continue
                        info(("{}| {} {}: " + self.print_precision).format(rtype, aggr, measure, self.performance[rtype][measure][aggr]["folds"][fold_index]))
            else:
                for measure in self.multilabel_measures:
                    container = self.performance[rtype][measure]
                    if not container:
                        continue
                    info(("{}| {}:" + self.print_precision).format(rtype, measure, self.performance[rtype][measure]["folds"][fold_index]))
        info("---------------")

    # compute scores and append to per-fold lists
    def add_performance(self, run_type, preds_proba):
        # loop thresholds & amax, get respective TPs, FPs, etc
        # evaluate metrics there, and multilabel evals with these.

        if self.num_labels != preds_proba.shape[-1]:
            error("Attempted to evaluated {}-dimensional predictions against {} labels".format(preds_proba.shape[-1], self.num_labels))
        if self.do_multilabel:
            onehot_gt = one_hot(self.test_labels, self.num_labels)

            # average precision
            ap = metrics.average_precision_score(onehot_gt, preds_proba)
            rocauc = metrics.roc_auc_score(onehot_gt, preds_proba)

            self.performance[run_type]["ap"]["folds"].append(ap)
            self.performance[run_type]["roc_auc"]["folds"].append(rocauc)
            return

        preds_amax = np.argmax(preds_proba, axis=1)
        # get prec, rec, f1
        for measure in [x for x in self.measures if x != "accuracy"]:
            cw, ma, mi, ws = self.get_pre_rec_f1(preds_amax, measure, self.num_labels)
            self.performance[run_type][measure]["classwise"]["folds"].append(cw)
            self.performance[run_type][measure]["macro"]["folds"].append(ma)
            self.performance[run_type][measure]["micro"]["folds"].append(mi)
            self.performance[run_type][measure]["weighted"]["folds"].append(ws)

        # get accuracies
        acc, cw_acc = self.compute_accuracy(preds_amax), self.compute_classwise_accuracy(preds_amax)
        self.performance[run_type]["accuracy"]["classwise"]["folds"].append(cw_acc)
        self.performance[run_type]["accuracy"]["macro"]["folds"].append(acc)

    # compute classification baselines
    def compute_performance(self, predictions):
        # # get multiclass performance
        # for av in ["macro", "micro"]:
        #     auc, ap = self.get_roc(predictions, average=av)
        #     self.performance["run"]["AP"][av] = ap
        #     self.performance["run"]["AUC"][av] = auc

        # compute single-label baselines
        # add run performance wrt argmax predictions
        self.add_performance("run", predictions)
        # majority classifier
        maxlabel = get_majority_label(self.test_labels, self.num_labels, self.do_multilabel)
        majpred = np.zeros(predictions.shape, np.float32)
        majpred[:, maxlabel] = 1.0
        self.add_performance("majority", majpred)
        # random classifier
        randpred = np.random.rand(*predictions.shape)
        self.add_performance("random", randpred)

    # applies the threshold to the probabilistic predictions, extracting decision indices
    def apply_decision_threshold(self, proba, thresh):
        decisions = []
        for row in proba:
            idxs = np.where(row > thresh)
            decisions.append(idxs)
        return decisions

    def get_score_stats(self, container):
        scores_str = []
        for stat in self.preferred_stats:
            value = container[stat]
            if type(value) == list:
                # folds
                scores_str.append("{" + " ".join(list(map(lambda x: self.print_precision.format(x), value))) + "}")
            else:
                scores_str.append(self.print_precision.format(value))
        return " ".join(scores_str)

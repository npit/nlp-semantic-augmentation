import pickle
import random
from os.path import join

import numpy as np
import pandas as pd
from sklearn import metrics

from utils import (count_label_occurences, debug, error, info,
                   numeric_to_string, one_hot, warning)


class Evaluator:
    """Class to produce evaluation measures from a learning run
    """

    # performance containers, in the format type-measure-aggregation-stat
    performance = None
    majority_label = None

    # predictions storage
    predictions = None
    predictions_instance_indexes = None
    confusion_matrices = None

    # available measures, aggregations and run types
    singlelabel_measures = ["precision", "recall", "f1-score", "accuracy", "ari", "shilhouette"]
    fold_aggregations = ["mean", "var", "std", "folds"]
    run_types = ["random", "dummy", "majority", "run"]
    multilabel_measures = ["ap", "roc_auc"]
    label_aggregations = ["macro", "micro", "classwise", "weighted"]

    # additional grouping
    unsupervised_measures = ["shilhouette"]
    supervised_types = ["dummy", "majority"]

    # defined in runtime
    measures = None

    # print opts
    print_precision = "{:.03f}"
    do_multilabel = None

    # validation opts
    test_via_validation = None
    # error analysis
    error_analysis = None
    error_print_types = None
    top_k = 3

    #  labels
    train_labels = None
    test_labels = None
    reference_labels = None

    def get_labelwise_measures(self):
        """Get measures definable in a multiclass setting"""
        return [x for x in self.measures if x not in ["ari"]]

    def has_test_labels(self):
        return self.test_labels is not None and len(self.test_labels) > 0

    def is_supervised(self):
        """Getter for supervised status"""
        return self.train_labels is not None or self.has_test_labels()

    # constructor
    def __init__(self, config, test_via_validation):
        """Evaluator constructor method"""
        self.config = config
        self.test_via_validation = test_via_validation
        # minor inits
        self.predictions = {rt: [] for rt in self.run_types}
        self.confusion_matrices = {rt: [] for rt in self.run_types}
        self.predictions_instance_indexes = []
        self.rename_micro_to_accuracy = False

    def get_current_reference_labels(self):
        return self.reference_labels[self.test_index]

    def update_reference(self, train_index, test_index, embeddings):
        if not self.test_via_validation:
            # if we have a dedicated test set, no need to update anything
            return
        # update train/test index instances
        self.train_index = train_index
        self.test_index = test_index
        self.embeddings = embeddings
        # if test_labels is not None:
        #     self.test_labels = test_labels
        #     if not self.do_multilabel and type(test_labels) is list:
        #         # squeeze to ndarray if necessary
        #         test_labels = np.squeeze(np.concatenate(test_labels))
        # if train_labels is not None:
        #     if not self.do_multilabel and type(test_labels) is list:
        #         test_labels = np.squeeze(np.concatenate(test_labels))
        #         train_labels = np.squeeze(np.concatenate(train_labels))
        #     self.train_labels = train_labels

    def show_reference_label_distribution(self):
        """Show distribution of reference labels
        """
        if self.is_supervised():
            if self.test_via_validation:
                self.show_label_distribution(self.train_labels, message="Overall training label distribution")
            else:
                self.show_label_distribution(self.test_labels, message="Test label distribution")

    def set_labelling(self, train_labels, num_labels, do_multilabel=False, test_labels=None):
        # train labels: should always be provided
        self.train_labels = train_labels
        if len(train_labels) == 0:
            error("Empty train labels container passed to the evaluator.")
        self.do_multilabel = do_multilabel
        self.num_labels = num_labels
        # squeeze to ndarray if necessary
        if not self.do_multilabel:
            if type(train_labels) is list:
                self.train_labels = np.squeeze(np.concatenate(train_labels))

        # test labels
        if test_labels is not None and len(test_labels)>0:
            self.test_labels = test_labels
            if type(test_labels) is list:
                self.test_labels = np.squeeze(np.concatenate(test_labels))

        if self.test_via_validation:
            self.reference_labels = self.train_labels
            # the train/test index will be updated via the update_reference function
        else:
            self.reference_labels = self.test_labels
            self.test_index = np.arange(len(self.test_labels))

    def check_setting(self, setting, available, what="configuration", adj="Erroneous", fatal=False):
        if not setting:
            return True
        clash = [x for x in setting if x not in available]
        msg = f"{adj} {what}(s): {clash} for current data and/or learner ({self.config.name}). "
        if clash:
            if fatal:
                msg += f"Use among: {available}"
                error(msg)
            msg += f"Defaulting to (preferred subset of): {available}"
            warning(msg)
            while setting:
                setting.pop()
            setting.extend(available)

    def check_sanity(self):
        """Function to check for invalid / incompatible configuration"""

        # output options
        self.preferred_types = self.config.print.run_types if self.config.print.run_types else self.run_types
        self.preferred_measures = self.config.print.measures if self.config.print.measures else []
        self.preferred_label_aggregations = self.config.print.label_aggregations if self.config.print.label_aggregations else self.label_aggregations
        self.preferred_fold_aggregations = self.config.print.fold_aggregations if self.config.print.fold_aggregations else self.fold_aggregations
        self.top_k = self.config.print.top_k
        error("Invalid value for top-k printing: {}".format(self.top_k), self.top_k <= 0)

        self.check_setting(self.preferred_types, self.run_types, "run type", "Unavailable", fatal=True)
        self.check_setting(self.preferred_measures, self.singlelabel_measures + self.multilabel_measures, "measure", "Unavailable", fatal=True)
        self.check_setting(self.preferred_label_aggregations, self.label_aggregations, "aggregation", "Unavailable", fatal=True)
        self.check_setting(self.preferred_fold_aggregations, self.fold_aggregations, "aggregation", "Unavailable", fatal=True)

        # restrict to compatible run types / measures
        # set measures type wrt labelling problem
        self.measures = self.multilabel_measures if self.do_multilabel else self.singlelabel_measures
        # set measures type wrt supervision
        if not self.is_supervised():
            self.run_types = [m for m in self.run_types if m not in self.supervised_types]
            self.measures = [m for m in self.measures if m in self.unsupervised_measures]
        else:
            self.measures = [m for m in self.measures if m not in self.unsupervised_measures]

        self.check_setting(self.preferred_types, self.run_types, "run type", "Incompatible")
        self.check_setting(self.preferred_measures, self.measures, "measure", "Incompatible")
        self.check_setting(self.preferred_label_aggregations, self.label_aggregations, "aggregation", "Incompatible")
        self.check_setting(self.preferred_fold_aggregations, self.fold_aggregations, "aggregation", "Unavailable")

        # init performance containers
        self.initialize_containers()

    # initialize evaluation containers and preferred evaluation printage
    def initialize_containers(self):
        """Method to initialize evaluation measure containers"""
        if self.performance is not None:
            return

        self.performance = {}
        for run_type in self.run_types:
            self.performance[run_type] = {}
            for measure in self.measures:
                self.performance[run_type][measure] = {}
                if self.do_multilabel:
                    for stat in self.fold_aggregations:
                        self.performance[run_type][measure][stat] = None
                    self.performance[run_type][measure]["folds"] = []
                else:
                    for aggr in self.label_aggregations:
                        self.performance[run_type][measure][aggr] = {}
                        for stat in self.fold_aggregations:
                            self.performance[run_type][measure][aggr][stat] = None
                        self.performance[run_type][measure][aggr]["folds"] = []

        # remove undefined combos
        if not self.do_multilabel:
            for run_type in self.run_types:
                for aggr in [x for x in self.label_aggregations if x not in ["macro", "classwise"]]:
                    if "accuracy" in self.performance[run_type]:
                        del self.performance[run_type]["accuracy"][aggr]

                for aggr in self.label_aggregations:
                    if aggr != "macro":
                        if "ari" in self.performance[run_type]:
                            del self.performance[run_type]["ari"][aggr]
                        if "shilhouette" in self.performance[run_type]:
                            del self.performance[run_type]["shilhouette"][aggr]

    def get_evaluation_axes_to_print(self):
        """Retrieve run types, measures and aggregations to display overall results"""
        res = {}
        for t in self.preferred_types:
            if self.do_multilabel:
                res[t] = set()
            else:
                res[t] = {}
            for m in self.preferred_measures:
                # make sure it's compatible with the run type
                if m not in self.performance[t]:
                    continue
                if self.do_multilabel:
                    res[t].add(m)
                else:
                    res[t][m] = set()
                if not self.do_multilabel:
                    for laggr in self.performance[t][m]:
                        if laggr not in self.performance[t][m]:
                            continue
                        res[t][m].add(laggr)
        return res



    # aggregated evaluation measure function shortcuts
    def get_pre_rec_f1(self, preds, metric, num_labels, gt=None):
        """Function to compute precision, recall and F1-score"""
        if gt is None:
            gt = self.get_current_reference_labels()
        # expected column structure is label1 label2 ... labelN microavg macroavg weightedavg
        cr = pd.DataFrame.from_dict(metrics.classification_report(gt, preds, output_dict=True))
        # get classwise, micro, macro, weighted, defaulting non-precited classes to zero values
        predicted_classes = cr.columns.to_list()[:-3]
        cw = np.zeros(num_labels, np.float32)
        for class_index in predicted_classes:
            cw[int(class_index)] = cr[class_index][metric]
        res = [cw]

        # gather the rest of the aggregations
        aggregations = ["macro avg", "weighted avg"]
        if "micro avg" in cr:
            aggregations = ["micro avg"] + aggregations
        else:
            aggregations = ["accuracy"] + aggregations
            debug("Micro-averaging mapped to accuracy in the confusion matrix.")
            self.rename_micro_to_accuracy = True

        for aggr in aggregations:
            try:
                r = cr[aggr][metric]
            except KeyError:
                warning("Failed to access {} avg information on the {} metric".format(aggr, metric))
                r = -1
            res.append(r)
        return res

    # adjusted rank index
    def compute_ari(self, preds, gt=None):
        if gt is None:
            gt = self.get_current_reference_labels()
        return metrics.adjusted_rand_score(gt, preds)

    # shilhouette coefficient
    def compute_shilhouette(self, preds):
        test_data = self.embeddings[self.test_index, :]
        return metrics.silhouette_score(test_data, preds)

    # get average accuracy
    def compute_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.get_current_reference_labels()
        return metrics.accuracy_score(gt, preds)

    # get class-wise accuracies
    def compute_classwise_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.get_current_reference_labels()
        cm = metrics.confusion_matrix(gt, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm.diagonal()

    def print_performance(self, run_type, measure, aggr=None):
        """performance printing function, checking respective settings and inputs for wether to print"""
        # print the combination, if it's in the prefered stuff to print
        if all([run_type in self.preferred_types, measure in self.preferred_measures, aggr is None or aggr in self.preferred_label_aggregations]):
            scores_str = self.get_score_stats_string(self.performance[run_type][measure][aggr])
            # print
            header = " ".join(["{:10}".format(x) for x in [run_type, aggr, measure] if x is not None])
            info("{} : {}".format(header, scores_str))
            return True
        return False

    # print performance across folds and compute foldwise aggregations
    def report_overall_results(self, validation_description, num_total_train, write_folder):
        """Function to report learning results
        """
        # compute overall predicted labels across all folds, if any
        overall_predictions = np.concatenate([np.argmax(x, axis=1) for x in self.predictions['run']])
        # count distribution occurences (validation total average)
        self.show_label_distribution(overall_predictions, message="Predicted label (average) distribution")
        info("==============================")
        self.analyze_overall_errors(num_total_train)
        info("==============================")

        info("Printing overall results:")
        info("Run types: {}".format("/".join(self.preferred_types)))
        info("Measures: {}".format("/".join(self.preferred_measures)))
        info("Label aggregations: {}".format("/".join(self.label_aggregations)))
        info("Fold aggregations: {}".format("/".join(self.fold_aggregations)))

        info("------------------------------")
        print_count = 0
        axes_dict = self.get_evaluation_axes_to_print()
        for run_type in axes_dict:
            for measure in axes_dict[run_type]:
                if self.do_multilabel:
                    self.performance[run_type][measure] = self.calc_fold_score_stats(self.performance[run_type][measure])
                    print_count += self.print_performance(run_type, measure)
                else:
                    for aggregation in axes_dict[run_type][measure]:
                        self.performance[run_type][measure][aggregation] = self.calc_fold_score_stats(self.performance[run_type][measure][aggregation])
                        print_count += self.print_performance(run_type, measure, aggregation)
            info("------------------------------")
        info("==============================")
        if not print_count:
            warning("No suitable run / measure /aggregation specified in the configuration.")

        if write_folder is not None:
            # write the results in csv in the results directory
            # entries in a run_type - measure configuration list are the foldwise scores, followed by the mean
            write_dict = {k: v for (k, v) in self.performance.items()}
            write_dict["error_analysis"] = self.error_analysis
            write_dict["confusion_matrices"] = self.confusion_matrices
            df = pd.DataFrame.from_dict(write_dict)
            df.to_csv(join(write_folder, "results.txt"))
            with open(join(write_folder, "results.pickle"), "wb") as f:
                pickle.dump({"results": df, "error_analysis": self.error_analysis, "confusion_matrix": self.confusion_matrices}, f)

    # print performance of the latest run
    def print_run_performance(self, current_run_descr, fold_index=0):
        info("---------------")
        info("Test results for {}:".format(current_run_descr))
        for rtype in self.preferred_types:
            if not self.do_multilabel:
                for measure in self.preferred_measures:
                    for aggr in self.preferred_label_aggregations:
                        # don't print classwise results or unedfined aggregations
                        if aggr not in self.performance[rtype][measure]:
                            continue
                        container = self.performance[rtype][measure][aggr]
                        if not container:
                            continue
                        str_value = numeric_to_string(self.performance[rtype][measure][aggr]["folds"][fold_index], self.print_precision)
                        info(("{}| {} {}: {}").format(rtype, aggr, measure, str_value))
            else:
                for measure in self.multilabel_measures:
                    container = self.performance[rtype][measure]
                    if not container:
                        continue
                    info(("{}| {}:" + self.print_precision).format(rtype, measure, self.performance[rtype][measure]["folds"][fold_index]))
        info("---------------")

    def evaluate_supervised(self, run_type, preds_proba):
        """Perform supervised evaluation of a run"""
        # sanity
        if self.num_labels != preds_proba.shape[-1]:
            error("Attempted to evaluated {}-dimensional predictions against {} labels".format(preds_proba.shape[-1], self.num_labels))

        if self.do_multilabel:
            onehot_gt = one_hot(self.get_current_reference_labels(), self.num_labels)

            # average precision
            ap = metrics.average_precision_score(onehot_gt, preds_proba)
            rocauc = metrics.roc_auc_score(onehot_gt, preds_proba)

            self.performance[run_type]["ap"]["folds"].append(ap)
            self.performance[run_type]["roc_auc"]["folds"].append(rocauc)
        else:
            # single-label
            preds_amax = np.argmax(preds_proba, axis=1)
            # get prec, rec, f1
            for measure in "precision recall f1-score".split():
                cw, mi, ma, we = self.get_pre_rec_f1(preds_amax, measure, self.num_labels)
                self.performance[run_type][measure]["classwise"]["folds"].append(cw)
                self.performance[run_type][measure]["macro"]["folds"].append(ma)
                self.performance[run_type][measure]["micro"]["folds"].append(mi)
                self.performance[run_type][measure]["weighted"]["folds"].append(we)
                self.confusion_matrices[run_type].append(metrics.confusion_matrix(self.get_current_reference_labels(), preds_amax, range(self.num_labels)))

            # get defined accuracy measures and aggregations
            acc, cw_acc = self.compute_accuracy(preds_amax), self.compute_classwise_accuracy(preds_amax)
            self.performance[run_type]["accuracy"]["classwise"]["folds"].append(cw_acc)
            self.performance[run_type]["accuracy"]["macro"]["folds"].append(acc)

            # get defined ari measures and aggregations
            self.performance[run_type]["ari"]["macro"]["folds"].append(self.compute_ari(preds_amax))

    def evaluate_unsupervised(self, run_type, preds_proba):
        """Evaluate unsupervised run"""
        preds_amax = np.argmax(preds_proba, axis=1)
        self.performance[run_type]["shilhouette"]["macro"]["folds"].append(self.compute_shilhouette(preds_amax))

    # compute scores and append to per-fold lists
    def evaluate_predictions(self, run_type, preds_proba):
        """Perform a run evaluation"""
        if self.is_supervised():
            self.evaluate_supervised(run_type, preds_proba)
        else:
            self.evaluate_unsupervised(run_type, preds_proba)

    def show_label_distribution(self, labels, message="Test label distribution:"):
        info("==============================")
        info(message + " (label, count)")
        info("------------------------------")
        distr = count_label_occurences(labels)
        local_max_lbl = max(distr, key=lambda x: x[1])[0]
        for lbl, freq in distr:
            maj = " - [input majority]" if lbl == self.majority_label else ""
            localmaj = " - [local majority]" if lbl == local_max_lbl else ""
            info("Label {} : {:.1f}{}{}".format(lbl, freq, localmaj, maj))

    # evaluate predictions and add baselines
    def evaluate_learning_run(self, predictions, instance_indexes=None):
        # store instance indexes
        if instance_indexes is None:
            # use the entire index set of the current test labels
            instance_indexes = np.asarray(list(range(len(self.get_current_reference_labels()))), np.int32)
        self.predictions_instance_indexes.append(instance_indexes)

        if len(predictions) != len(self.test_index):
            error("Inconsistent shapes of predictions: {} and labels: {} lengths during evaluation"
                  .format(len(predictions), len(self.test_index)))
        # compute single-label baselines
        # add run performance wrt argmax predictions
        self.evaluate_predictions("run", predictions)
        self.predictions["run"].append(predictions)

        # majority classifier - always predicts the majority label
        if self.is_supervised():
            # majority learner
            if self.majority_label is None:
                self.majority_label = count_label_occurences(self.get_current_reference_labels(), return_only_majority=True)
            info("Majority label: {}".format(self.majority_label))
            majpred = np.zeros(predictions.shape, np.float32)
            majpred[:, self.majority_label] = 1.0
            self.evaluate_predictions("majority", majpred)
            self.predictions["majority"].append(majpred)

            # dummy learner - predicts wrt the training label distribution
            # handle cases with a larger training set
            num_test, num_train = len(instance_indexes), len(self.train_labels)
            num_tile = np.ceil(num_test / num_train)
            sample_pool = np.tile(self.train_labels, int(num_tile))[:num_test]
            dummypred = one_hot(random.sample(list(sample_pool), num_test), self.num_labels)
            self.evaluate_predictions("dummy", dummypred)
            self.predictions["dummy"].append(dummypred)

        # random classifier
        randpred = np.random.rand(*predictions.shape)
        self.evaluate_predictions("random", randpred)
        self.predictions["random"].append(randpred)

    # applies the threshold to the probabilistic predictions, extracting decision indices
    def apply_decision_threshold(self, proba, thresh):
        decisions = []
        for row in proba:
            idxs, _ = np.where(row > thresh)
            decisions.append(idxs)
        return decisions

    # compute statistics across folds
    def calc_fold_score_stats(self, container):
        # set foldwise scores to an nparray
        container["folds"] = np.asarray(container["folds"])
        for key, func in [("mean", np.mean), ("var", np.var), ("std", np.std)]:
            container[key] = func(container["folds"], axis=0)
        return container

    # get a printable format of evaluation
    def get_score_stats_string(self, container):
        scores_str = []
        for stat in self.preferred_fold_aggregations:
            scores_str.append(numeric_to_string(container[stat], self.print_precision))
        return " ".join(scores_str)

    def consolidate_folded_test_results(self, num_total_train):
        for run_type in self.predictions:
            for p, preds in enumerate(self.predictions[run_type]):
                full_preds = np.zeros((num_total_train, self.num_labels), np.float32)
                full_preds[:] = np.nan
                idxs = self.predictions_instance_indexes[p]
                full_preds[idxs] = preds
                self.predictions[run_type][p] = full_preds

    # analyze overall error of the run
    def analyze_overall_errors(self, num_total_train):
        """Method to generate an error analysis over folds
        - label-wise ranking: extract best/worst labels, average across instances
        - instance-wise ranking: extract best/worst instances, average across labels
          - run type (e.g. regular, random, majority)
          - evaluation measure (e.g. accuracy, f1)
          - fold-wise aggregation evaluation measure stat (mean, std, variance)

        for labels: use classwise, mean per fold
        for instances: have to save test predictions, mean per fold
        for across runs, have to use large_run externally


        - print top / bottom K
        """
        if not self.is_supervised():
            return
        if not self.do_multilabel:
            res = {"instances": {}, "labels": {}}

            if self.test_via_validation:
                # instances vary across folds -- consolidate
                self.consolidate_folded_test_results(num_total_train)

            for run_type in self.predictions:
                # instance-wise
                aggregate = np.zeros((len(self.get_current_reference_labels()),), np.float32)
                for preds in self.predictions[run_type]:
                    # get true / false predictions
                    non_nan_idx = np.where(np.any(~np.isnan(preds), axis=1))
                    preds = preds[non_nan_idx]
                    true_labels = np.squeeze(self.reference_labels[non_nan_idx])
                    correct_preds = np.where(np.equal(np.argmax(preds, axis=1), true_labels))
                    aggregate[correct_preds] += 1
                # average
                aggregate /= len(self.predictions[run_type])
                # sort scores and instance indexes
                ranked_scores_idxs = sorted(list(zip(aggregate, list(range(len(aggregate))))), key=lambda x: x[0], reverse=True)
                res["instances"][run_type] = ranked_scores_idxs

                # label-wise
                res["labels"][run_type] = {}
                for measure in self.get_labelwise_measures():
                    if "classwise" in self.performance[run_type][measure]:
                        scores_cw = self.performance[run_type][measure]['classwise']['folds']
                        # average accross folds
                        scores_cw = sum(scores_cw) / len(scores_cw)
                        # rank
                        ranked_scores_idxs = sorted(list(zip(scores_cw, list(range(len(scores_cw))))), key=lambda x: x[0], reverse=True)
                        res["labels"][run_type][measure] = ranked_scores_idxs
        else:
            pass

        self.error_analysis = res
        self.print_error_analysis()

    def print_error_analysis(self):
        """Function that outputs the computed error analysis
        """
        if not self.config.print.error_analysis:
            return
        info("Fold-average top-{} instances:".format(self.top_k))
        info("---------------------------")
        for run_type in self.preferred_types:
            # print in format instance1, instance2, ...
            # print the below error / stat visualization
            print_types = {"top": lambda x: x[:self.top_k], "bottom": lambda x: x[-self.top_k:]}
            for print_type, func in print_types.items():
                indexes = " ".join("{:.0f}".format(x[1]) for x in func(self.error_analysis["instances"][run_type]))
                scores = " ".join("{:1.3f}".format(x[0]) for x in func(self.error_analysis["instances"][run_type]))
                info("{:10} {:8} {:7} {:10} | ({}) ({})".format("accuracy", run_type, print_type, "instances", indexes, scores))

        lbl_top = min(self.top_k, self.num_labels)
        info("---------------------------")
        info("Fold-average top-{} labels:".format(lbl_top))
        info("---------------------------")
        for run_type in self.preferred_types:
            print_types = {"top": lambda x: x[:lbl_top], "bottom": lambda x: x[-lbl_top:]}
            for print_type, func in print_types.items():
                for measure in [x for x in self.performance[run_type] if x in self.preferred_measures]:
                    info("{:10} {:8} {:7} {:10} | ({}) ({})".format(measure, run_type, print_type, "labels",
                                                                    " ".join("{:.0f}".format(x[1]) for x in func(self.error_analysis["labels"][run_type][measure])),
                                                                    " ".join("{:1.3f}".format(x[0]) for x in func(self.error_analysis["labels"][run_type][measure]))))

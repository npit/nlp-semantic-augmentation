from learner import Learner
import learner
from utils import tictoc, write_pickled, info, error, read_pickled
from os.path import join, dirname, exists, basename
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from os import makedirs


class Classifier(Learner):

    def __init__(self):
        """Generic classifier constructor
        """
        Learner.__init__(self)

    def make(self, representation, dataset):
        Learner.make(self, representation, dataset)

    # perfrom a train-test loop
    def do_traintest(self):
        # get trainval data
        train_val_idxs = self.get_trainval_indexes()

        # keep track of models' test performances and paths wrt selected metrics
        model_paths = []

        with tictoc("Total training", do_print=self.do_folds, announce=False):
            # loop on folds, or do a single loop on the train-val portion split
            for fold_index, trainval_idx in enumerate(train_val_idxs):
                self.fold_index = fold_index
                if self.do_folds:
                    self.current_run_descr = "fold {}/{}".format(fold_index + 1, self.folds)
                elif self.do_validate_portion:
                    self.current_run_descr = "{}-val split".format(self.validation_portion)
                else:
                    self.current_run_descr = "(no-validation)"

                # check if the run is completed already, if allowed
                if not self.forbid_load:
                    existing_predictions = self.is_already_completed()
                    if existing_predictions is not None:
                        self.evaluator.compute_performance(existing_predictions)
                        continue
                # train the model
                with tictoc("Training run {} on train/val data :{}.".format(self.current_run_descr, list(map(len, trainval_idx)))):
                    model = self.train_model(trainval_idx)

                # test the model
                with tictoc("Testing {} on data: {}.".format(self.current_run_descr, self.num_test_labels)):
                    self.do_test(model)
                    model_paths.append(self.model_saver.filepath)

            self.evaluator.report_results(self.folds, self.results_folder)

    # evaluate a model on the test set
    def do_test(self, model):
        print_results = self.do_folds and self.config.print.folds or not self.folds
        test_data = self.process_input(self.test)
        predictions = model.predict(test_data, batch_size=self.batch_size, verbose=self.verbosity)
        # get baseline performances
        self.evaluator.compute_performance(predictions)
        if print_results:
            self.evaluator.print_performance(self.current_run_descr, self.fold_index)
        # write fold predictions
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        write_pickled(predictions_file, predictions)

    # produce training / validation splits, with respect to sample indexes
    def get_trainval_indexes(self):
        if not self.validation_exists:
            return [(np.arange(self.num_train_labels), np.arange(0))]

        trainval_serialization_file = join(self.results_folder, basename(self.get_current_model_path()) + ".trainval.pickle")
        if self.do_folds:
            # check if such data exists
            if exists(trainval_serialization_file) and not self.forbid_load:
                info("Training {} with input data: {} samples, {} labels, on LOADED existing {} stratified folds".format(
                    self.name, self.num_train, self.num_train_labels, self.folds))
                deser = read_pickled(trainval_serialization_file)
                if not len(deser) == self.folds:
                    error("Mismatch between expected folds ({}) and loaded data of {} splits.".format(self.folds, len(deser)))
                max_idx = max([np.max(x) for tup in deser for x in tup])
                if max_idx >= self.num_train:
                    error("Mismatch between max instances in training data ({}) and loaded max index ({}).".format(self.num_train, max_idx))
                return deser
            info("Training {} with input data: {} samples, {} labels, on {} stratified folds".format(
                self.name, self.num_train, self.num_train_labels, self.folds))
            # for multilabel K-fold, stratification is not available
            FoldClass = KFold if self.do_multilabel and self.do_folds else StratifiedKFold
            splitter = FoldClass(self.folds, shuffle=True, random_state=self.seed)

        if self.do_validate_portion:
            # check if such data exists
            if exists(trainval_serialization_file) and not self.forbid_load:
                info("Training {} with input data: {} samples, {} labels, on LOADED existing {} validation portion".format(self.name, self.num_train, self.num_train_labels, self.validation_portion))
                deser = read_pickled(trainval_serialization_file)
                info("Loaded train/val split of {} / {}.".format(*list(map(len, deser[0]))))
                # sanity checks
                max_idx = max([np.max(x) for tup in deser for x in tup])
                if max_idx >= self.num_train:
                    error("Mismatch between max instances in training data ({}) and loaded max index ({}).".format(self.num_train, max_idx))
                return deser
            info("Splitting {} with input data: {} samples, {} labels, on a {} validation portion".format(self.name, self.num_train, self.num_train_labels, self.validation_portion))
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_portion, random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(splitter.split(np.zeros(self.num_train_labels), self.train_labels))
        # save and return the splitter splits
        makedirs(dirname(trainval_serialization_file), exist_ok=True)
        write_pickled(trainval_serialization_file, splits)
        return splits

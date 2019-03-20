from utils import error, info, read_pickled, tictoc, write_pickled, one_hot
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from os.path import join, dirname, exists, basename
from evaluator import Evaluator
from os import makedirs


class Learner:

    save_dir = "models"
    folds = None
    fold_index = 0
    evaluator = None
    sequence_length = None

    def __init__(self):
        """Generic learner constructor
        """
        # initialize evaluation
        self.evaluator = Evaluator(self.config)
        pass

    # input preproc
    def process_input(self, data):
        return data

    def make(self, representation, dataset):
        self.verbosity = 1 if self.config.print.training_progress else 0
        # get data and labels
        self.train, self.test = representation.get_data()
        self.train_labels, self.test_labels = [x for x in dataset.get_labels()]
        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(zero_samples_idx))
        # nan checks
        nans = np.where(np.isnan(self.train))
        if np.size(nans) != 0:
            error("NaNs in training data:{}".format(nans))
        nans = np.where(np.isnan(self.test))
        if np.size(nans) != 0:
            error("NaNs in test data:{}".format(nans))

        # get many handy variables
        self.do_multilabel = dataset.is_multilabel()
        self.num_labels = dataset.get_num_labels()
        self.num_train, self.num_test, self.num_train_labels, self.num_test_labels = \
            list(map(len, [self.train, self.test, self.train_labels, self.test_labels]))
        self.evaluator.set_labels(self.test_labels, self.num_labels)
        self.input_dim = representation.get_dimension()
        self.forbid_load = self.config.learner.no_load
        self.sequence_length = self.config.learner.sequence_length
        self.results_folder = self.config.folders.results
        self.models_folder = join(self.results_folder, "models")
        self.epochs = self.config.train.epochs
        self.folds = self.config.train.folds
        self.validation_portion = self.config.train.validation_portion
        self.do_folds = self.folds and self.folds > 1
        self.do_validate_portion = self.validation_portion is not None and self.validation_portion > 0.0
        self.validation_exists = self.do_folds or self.do_validate_portion
        self.early_stopping_patience = self.config.train.early_stopping_patience

        self.seed = self.config.get_seed()
        np.random.seed(self.seed)

        self.batch_size = self.config.train.batch_size
        info("Learner data/labels: train: {} test: {}".format(self.train.shape, self.test.shape))

        # sanity checks
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(self.folds, self.validation_portion))

        # measure sanity checks
        self.evaluator.check_sanity(self.do_multilabel)

    def is_already_completed(self):
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        if exists(predictions_file):
            info("Reading existing predictions: {}".format(predictions_file))
            return read_pickled(predictions_file)
        return None

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
                    model_paths.append(self.get_current_model_path())

            self.evaluator.report_results(self.folds, self.results_folder)

    # evaluate a model on the test set
    def do_test(self, model):
        print_results = self.do_folds and self.config.print.folds or not self.folds
        test_data = self.process_input(self.test)
        predictions = self.test_model(test_data, model)
        # get baseline performances
        self.evaluator.compute_performance(predictions)
        if print_results:
            self.evaluator.print_performance(self.current_run_descr, self.fold_index)
        # write fold predictions
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        write_pickled(predictions_file, predictions)

    def get_current_model_path(self):
        filepath = join(self.models_folder, "{}".format(self.name))
        if self.do_folds:
            filepath += "_fold{}".format(self.fold_index)
        if self.do_validate_portion:
            filepath += "_valportion{}".format(self.validation_portion)
        return filepath

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

    # handle multi-vector items, expanding indexes to the specified sequence length
    def expand_index_to_sequence(self, fold_data):
        # map to indexes in the full-sequence data (e.g. times sequence_length)
        fold_data = list(map(lambda x: x * self.sequence_length if len(x) > 0 else np.empty((0,)), fold_data))
        for i in range(len(fold_data)):
            if fold_data[i] is None:
                continue
            # expand with respective sequence members (add an increment, vstack)
            stacked = np.vstack([fold_data[i] + incr for incr in range(self.sequence_length)])
            # reshape to a single vector, in the vertical (column) direction, that increases incrementally
            fold_data[i] = np.ndarray.flatten(stacked, order='F')
        return fold_data

    # split train/val labels and convert to one-hot
    def prepare_labels(self, trainval_idx):
        train_idx, val_idx = trainval_idx
        train_labels = self.train_labels
        if len(train_idx) > 0:
            train_labels = [self.train_labels[i] for i in train_idx]
            train_labels = one_hot(train_labels, self.num_labels)
        else:
            train_labels = np.empty((0,))
        if len(val_idx) > 0:
            val_labels = [self.train_labels[i] for i in val_idx]
            val_labels = one_hot(val_labels, self.num_labels)
        else:
            val_labels = np.empty((0,))
        return train_labels, val_labels

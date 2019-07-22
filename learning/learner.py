from bundle.bundle import BundleList, Bundle
from bundle.datatypes import Vectors, Labels
from component.component import Component
from utils import error, info, read_pickled, tictoc, write_pickled, one_hot, warning, is_multilabel, count_label_occurences
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from os.path import join, dirname, exists, basename
from learning.evaluator import Evaluator
from os import makedirs


"""
Abstract class representing a learning model
"""

class Learner(Component):

    component_name = "learner"
    save_dir = "models"
    folds = None
    fold_index = 0
    evaluator = None
    sequence_length = None
    train, test, train_labels, test_labels = None, None, None, None

    test_instance_indexes = None
    validation = None

    allow_learning_loading = None
    allow_prediction_loading = None

    def __init__(self):
        """Generic learning constructor
        """
        # initialize evaluation
        Component.__init__(self, consumes=[Vectors.name, Labels.name])
        self.can_be_final = True
        self.evaluator = Evaluator(self.config)

    # input preproc
    def process_input(self, data):
        return data

    def count_samples(self):
        self.num_train, self.num_test, self.num_train_labels, self.num_test_labels = \
            map(len, [self.train, self.test, self.train_labels, self.test_labels])

    def make(self):
        self.verbosity = 1 if self.config.print.training_progress else 0
        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(zero_samples_idx))
        # nan checks
        nans, _ = np.where(np.isnan(self.train))
        if np.size(nans) != 0:
            error("NaNs in training data:{}".format(nans))
        nans, _ = np.where(np.isnan(self.test))
        if np.size(nans) != 0:
            error("NaNs in test data:{}".format(nans))

        # get many handy variables
        self.do_multilabel = is_multilabel(self.train_labels)
        label_counts = count_label_occurences(self.train_labels)
        self.num_labels = len(label_counts)
        self.count_samples()
        self.input_dim = self.train[0].shape[-1]
        self.allow_prediction_loading = self.config.misc.allow_prediction_loading
        self.allow_learning_loading = self.config.misc.allow_learning_loading
        self.sequence_length = self.config.learner.sequence_length
        self.results_folder = self.config.folders.results
        self.models_folder = join(self.results_folder, "models")
        self.epochs = self.config.train.epochs
        self.folds = self.config.train.folds
        self.validation_portion = self.config.train.validation_portion
        self.do_folds = self.folds and self.folds > 1
        self.do_validate_portion = self.validation_portion is not None and self.validation_portion > 0.0
        self.validation_exists = (self.do_folds or self.do_validate_portion)
        self.use_validation_for_training = self.validation_exists and self.test_data_available()
        self.early_stopping_patience = self.config.train.early_stopping_patience

        self.validation = Learner.ValidatonSetting(self.folds, self.validation_portion, self.test_data_available(), self.do_multilabel)
        self.validation.assign_data(self.train, self.train_labels, self.test, self.test_labels)

        self.seed = self.config.get_seed()
        np.random.seed(self.seed)

        self.batch_size = self.config.train.batch_size
        info("Learner data/labels: train: {} test: {}".format(self.train.shape, self.test.shape))

        # sanity checks
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(self.folds, self.validation_portion))
        if not (self.validation_exists or self.test_data_available()):
            error("No test data or cross/portion-validation setting specified.")

        # configure and sanity-check evaluator
        if self.validation_exists and not self.use_validation_for_training:
            # calculate the majority label from the training data -- label counts already computed
            self.evaluator.majority_label = label_counts[0][0]
            info("Majority label: {}".format(self.evaluator.majority_label))
            self.evaluator.configure(self.train_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training, self.validation_exists)
            self.evaluator.compute_label_distribution()
            # self.evaluator.show_label_distribution(labels=self.train_labels, do_show=False)
        else:
            # count label distribution from majority
            self.evaluator.configure(self.test_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training, self.validation_exists)
            self.evaluator.majority_label = count_label_occurences(self.test_labels)[0][0]

        error("Input none dimension.", self.input_dim is None)
        info("Created learning: {}".format(self))

    def get_existing_predictions(self):
        path = self.validation.modify_suffix(join(self.results_folder, "{}".format(self.name))) + ".predictions.pickle"
        return read_pickled(path) if exists(path) else None

    def get_existing_trainval_indexes(self):
        """Check if the current training run is already completed."""
        trainval_file = self.get_trainval_serialization_file()
        if exists(trainval_file):
            info("Training {} with input data: {} samples, {} labels, on LOADED existing {}".format(self.name, self.num_train, self.num_train_labels, self.validation))
            idx = read_pickled(trainval_file)
            self.validation.check_indexes(idx)
            max_idx = max([np.max(x) for tup in idx for x in tup])
            if max_idx >= self.num_train:
                error("Mismatch between max instances in training data ({}) and loaded max index ({}).".format(self.num_train, max_idx))

    def get_existing_model_path(self):
        path = self.validation.modify_suffix(join(self.results_folder, "models", "{}".format(self.name))) + ".model"
        return path if exists(path) else None

    def test_data_available(self):
        return self.test.size > 0

    # function to retrieve training data as per the existing configuration
    def get_trainval_indexes(self):
        trainval_idx = None
        # get training / validation indexes
        if self.allow_learning_loading:
            ret = self.get_existing_run_data()
            if ret:
                trainval_idx, self.existing_model_paths = ret
        if not trainval_idx:
            trainval_idx = self.compute_trainval_indexes()

        # handle indexes for multi-instance data
        if self.num_train != self.num_train_labels:
            trainval_idx = self.expand_index_to_sequence(trainval_idx)
        return trainval_idx

    # perfrom a train-test loop
    def do_traintest(self):
        with tictoc("Entire learning run", do_print=self.do_folds, announce=False):

            # get trainval data
            train_val_idxs = self.get_trainval_indexes()

            # keep track of models' test performances and paths wrt selected metrics
            model_paths = []

            # iterate over required runs as per the validation setting
            for iteration_index, trainval in enumerate(train_val_idxs):
                train, val, test, test_instance_indexes = self.validation.get_run_data(iteration_index, trainval)

                # show training data statistics
                self.evaluator.show_label_distribution(count_label_occurences(train[1]), "Training label distr:" +  self.validation.get_current_descr())
                if val:
                    self.evaluator.show_label_distribution(count_label_occurences(val[1]), "Validation label distr:" +  self.validation.get_current_descr())

                # preprocess data and labels
                train, val, test = self.preprocess_data_labels(train, val, test)
                # assign containers
                train_data, train_labels = train
                test_data, test_labels = test
                val_data, val_labels = val if val else (None, None)

                # check if the run is completed already and load existing results, if allowed
                predictions = None
                if self.allow_prediction_loading:
                    # get predictions and instance indexes they correspond to
                    existing_preds, existing_instance_indexes = self.get_existing_predictions()
                    error("Different instance indexes loaded than the ones generated.", existing_instance_indexes != test_instance_indexes)
                    existing_test_labels = self.validation.get_test_labels(test_instance_indexes)
                    error("Different instance labels loaded than the ones generated.", existing_test_labels != test_labels)
                    predictions = existing_instance_indexes

                # train the model
                with tictoc("Training run {} on train data: {} and val data: {}.".format(self.validation, len(train_labels), len(val[1]) if val else "none")):
                    # check if a trained model already exists
                    model = None
                    if self.allow_learning_loading:
                        path = self.get_existing_model_path()
                        if path:
                            model = self.load_model(path)
                    if not model:
                        model = self.train_model(train_data, train_labels, val_data, val_labels)

                    if self.validation_exists:
                        self.evaluator.set_fold_info(train_labels)

                # test the model
                with tictoc("Testing {} on {} instances.".format(self.validation.descr, self.num_test_labels)):
                    self.do_test(model, test_data, test_labels, test_instance_indexes, predictions)
                    model_paths.append(self.get_current_model_path())

                if self.validation_exists and not self.use_validation_for_training:
                    self.test, self.test_labels = [], []
                    self.test_instance_indexes = None

                # wrap up validation iteration
                self.validation.conclude_iteration()

            if self.validation.use_validation_for_testing:
                # for the final evaluation, pass the entire training labels
                self.evaluator.configure(self.train_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training, self.validation_exists)
            else:
                # show test label distribution
                self.evaluator.show_label_distribution()
            self.evaluator.report_overall_results(self.validation.descr, len(self.train), self.results_folder)

    # evaluate a model on the test set
    def do_test(self, model, test_data, test_labels, test_instance_indexes, predictions=None):
        if not predictions:
            # evaluate the model
            info("Test data {}".format(test_data.shape))
            error("No test data supplied!", len(test_data) == 0)
            predictions = self.test_model(test_data, model)
        # get baseline performances
        self.evaluator.update_reference_labels(test_labels)
        self.evaluator.evaluate_learning_run(predictions, test_instance_indexes)
        if self.do_folds and self.config.print.folds:
            self.evaluator.print_run_performance(self.validation.descr, self.validation.current_fold)
        # write fold predictions
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        write_pickled(predictions_file, [predictions, test_instance_indexes])

    def get_current_model_path(self):
        filepath = join(self.models_folder, "{}".format(self.name))
        if self.do_folds:
            filepath += "_fold{}".format(self.fold_index)
        if self.do_validate_portion:
            filepath += "_valportion{}".format(self.validation_portion)
        return filepath

    def get_trainval_serialization_file(self):
        return join(self.results_folder, basename(self.get_current_model_path()) + ".trainval.pickle")

    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):
        if not self.validation_exists:
            return [(np.arange(self.num_train_labels), np.arange(0))]

        trainval_serialization_file = self.get_trainval_serialization_file()

        if self.do_folds:
            info("Training {} with input data: {} samples, {} labels, on {} stratified folds".format(
                self.name, self.num_train, self.num_train_labels, self.folds))
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
            else:
                splitter = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        if self.do_validate_portion:
            info("Splitting {} with input data: {} samples, {} labels, on a {} validation portion".format(self.name, self.num_train, self.num_train_labels, self.validation_portion))
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_portion, random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(splitter.split(np.zeros(self.num_train_labels), self.train_labels))
        # save and return the splitter splits
        makedirs(dirname(trainval_serialization_file), exist_ok=True)
        write_pickled(trainval_serialization_file, splits)
        return splits

    # apply required preprocessing on data and labels for a run
    def preprocess_data_labels(self, train, val, test):
        # since validation labels may be used for testing,
        # one-hot label computation is delayed until it's required
        train_data, train_labels = train
        # train_labels = one_hot(train_labels, self.num_labels)
        train = np.vstack(self.process_input(train_data)), train_labels

        if val is not None:
            val_data, val_labels = val
            # val_labels = one_hot(val_labels, self.num_labels)
            val = np.vstack(self.process_input(val_data)), val_labels

        test_data, test_labels = test
        test = self.process_input(test_data), test_labels
        return train, val, test

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

    def save_model(self):
        error("Attempted to access base save model function")
    def load_model(self):
        error("Attempted to access base load model function")

    # region: component functions
    def run(self):
        self.process_component_inputs()
        self.make()
        self.do_traintest()

        self.outputs.set_vectors(Vectors(vecs=self.evaluator.predictions))

    def process_component_inputs(self):
        # get data and labels
        error("Learner needs at least two-input bundle input list.", type(self.inputs) is not BundleList)
        error("{} needs vector information.".format(self.component_name), not self.inputs.has_vectors())
        error("{} needs label information.".format(self.component_name), not self.inputs.has_labels())

        self.train, self.test = self.inputs.get_vectors(single=True).instances
        self.train_labels, self.test_labels = self.inputs.get_labels(single=True).instances


    class ValidatonSetting:
        def __init__(self, folds, portion, test_present, do_multilabel):
            self.do_multilabel = do_multilabel
            self.do_folds = folds is not None
            self.folds = folds
            self.portion = portion
            self.do_portion = portion is not None
            self.use_validation_for_testing = not test_present
            if self.do_folds:
                self.descr = " {} stratified folds".format(self.folds)
                self.current_fold = 0
            elif self.do_portion:
                self.descr = "{} validation portion".format(self.portion)
            else:
                self.descr = "(no validation)"

        def __str__(self):
            return self.descr

        def get_current_descr(self):
            if self.do_folds:
                return "fold {}/{}".format(self.current_fold + 1, self.folds)
            elif self.do_portion:
                return "{}-val split".format(self.portion)
            else:
                return "(no-validation)"

        def check_indexes(self, idx):
            if self.do_folds:
                error("Mismatch between expected folds ({}) and loaded data of {} splits.".format(self.folds, len(idx)), len(idx) != self.folds)
            elif self.do_portion:
                info("Loaded train/val split of {} / {}.".format(*list(map(len, idx[0]))))

        def get_model_path(self, base_path):
            return self.modify_suffix(base_path) + ".model"

        def conclude_iteration(self):
            if self.do_folds:
                self.current_fold += 1

        def modify_suffix(self, base_path):
            if self.do_folds:
                return base_path + "fold{}.model".format(self.current_fold)
            elif self.do_portion:
                base_path += "valportion{}.model".format(self.portion)
            return base_path

        def assign_data(self, train, train_labels, test, test_labels):
            self.train = train
            self.train_labels = train_labels
            self.test = test
            self.test_labels = test_labels

        # get training, validation, test data chunks, given the input indexes and validation setting
        def get_run_data(self, iteration_index, trainval_idx):
            """get training and validation data chunks, given the input indexes"""
            if self.do_folds:
                error("Iteration index: {} / fold index: {} mismatch in validation coordinator.".format(iteration_index, self.current_fold), iteration_index != self.current_fold)
            train_idx, val_idx = trainval_idx
            if len(train_idx) > 0:
                train_labels = [np.asarray(self.train_labels[i]) for i in train_idx]
                if not self.do_multilabel:
                    train_labels = np.asarray(train_labels)
                train_data = np.vstack([self.train[idx] for idx in train_idx])
            else:
                train_data, train_labels = np.empty((0,)), np.empty((0))

            if len(val_idx) > 0:
                val_labels = [np.asarray(self.train_labels[i]) for i in val_idx]
                if not self.do_multilabel:
                    val_labels = np.asarray(val_labels)

                val_data = np.vstack([self.train[idx] for idx in val_idx])
                val = val_data, val_labels
            else:
                val = None

            if self.use_validation_for_testing:
                val, test = None, val
            else:
                test = self.test, self.test_labels
            if  len(val_idx) > 0 and self.use_validation_for_testing:
                # mark the test instance indexes as the val. indexes of the train
                instance_indexes = val_idx
            else:
                instance_indexes = range(len(test))
            return (train_data, train_labels), val, test, instance_indexes

        def get_test_labels(self, instance_indexes):
            if self.use_validation_for_testing:
                return self.train_labels[instance_indexes]
            else:
                error("Non-full instance indexes encountered, but validation is not set to act as testing",
                      instance_indexes != range(len(self.test_labels)))
                return self.test_labels


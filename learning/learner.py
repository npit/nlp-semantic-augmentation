from bundle.bundle import BundleList, Bundle
from bundle.datatypes import Vectors, Labels
from component.component import Component
from utils import error, info, read_pickled, tictoc, write_pickled, one_hot, warning, get_majority_label, is_multilabel
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
        nans = np.where(np.isnan(self.train))
        if np.size(nans) != 0:
            error("NaNs in training data:{}".format(nans))
        nans = np.where(np.isnan(self.test))
        if np.size(nans) != 0:
            error("NaNs in test data:{}".format(nans))

        # get many handy variables
        self.do_multilabel = is_multilabel(self.train_labels)
        label_counts = get_majority_label(self.train_labels, return_counts=True)
        self.num_labels = len(label_counts)
        self.count_samples()
        self.input_dim = self.train[0].shape[-1]
        self.predictions_loading_allowed = self.config.misc.predictions_loading_allowed
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
        self.evaluator.configure(self.test_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training)
        if self.validation_exists and not self.use_validation_for_training:
            # calculate the majority label from the training data -- label counts already computed
            self.evaluator.majority_label = label_counts[0][0]
            info("Majority label: {}".format(self.evaluator.majority_label))
            self.evaluator.show_label_distribution(labels=self.train_labels, do_show=False)

        error("Input none dimension.", self.input_dim is None)
        info("Created learning: {}".format(self))

    def is_already_completed(self):
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        if exists(predictions_file):
            warning("Reading existing predictions from: {}".format(predictions_file))
            return read_pickled(predictions_file)
        return None, None

    def test_data_available(self):
        return self.test.size > 0

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
                    validation_description = "folds={}".format(self.folds)
                elif self.do_validate_portion:
                    self.current_run_descr = "{}-val split".format(self.validation_portion)
                    validation_description = "validation portion={}".format(self.validation_portion)
                else:
                    self.current_run_descr = "(no-validation)"
                    validation_description = "<none>"

                # check if the run is completed already and load existing results, if allowed
                if self.predictions_loading_allowed:
                    existing_predictions, test_instance_indexes = self.is_already_completed()
                    if existing_predictions is not None:
                        if not self.use_validation_for_training:
                            labels = np.concatenate(self.train_labels)[test_instance_indexes]
                            self.evaluator.configure(labels, self.num_labels, self.do_multilabel, self.use_validation_for_training)
                        self.evaluator.evaluate_learning_run(existing_predictions, instance_indexes=test_instance_indexes)
                        continue
                # if no test data is available, use the validation data
                if self.validation_exists and not self.use_validation_for_training:
                    train_idx, val_idx = trainval_idx
                    self.test, self.test_labels = self.train[val_idx], self.train_labels[val_idx]
                    self.evaluator.configure(self.test_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training)
                    self.test_instance_indexes = val_idx
                    self.count_samples()
                    trainval_idx = (train_idx, [])

                # train the model
                with tictoc("Training run {} on train/val data :{}.".format(self.current_run_descr, list(map(len, trainval_idx)))):
                    model = self.train_model(trainval_idx)

                # test the model
                with tictoc("Testing {} on {} instances.".format(self.current_run_descr, self.num_test_labels)):
                    self.do_test(model)
                    model_paths.append(self.get_current_model_path())

                if self.validation_exists and not self.use_validation_for_training:
                    self.test, self.test_labels = [], []
                    self.test_instance_indexes = None

            if self.validation_exists and not self.use_validation_for_training:
                # pass the entire training labels
                self.evaluator.configure(self.train_labels, self.num_labels, self.do_multilabel, self.use_validation_for_training)
            self.evaluator.report_overall_results(validation_description, len(self.train), self.results_folder)

    # evaluate a model on the test set
    def do_test(self, model):
        print_results = self.do_folds and self.config.print.folds or not self.folds
        test_data = self.process_input(self.test)
        info("Test data {}".format(test_data.shape))
        error("No test data supplied!", len(test_data) == 0)
        predictions = self.test_model(test_data, model)
        # get baseline performances
        self.evaluator.evaluate_learning_run(predictions, self.test_instance_indexes)
        if print_results:
            self.evaluator.print_run_performance(self.current_run_descr, self.fold_index)
        # write fold predictions
        predictions_file = join(self.results_folder, basename(self.get_current_model_path()) + ".predictions.pickle")
        write_pickled(predictions_file, [predictions, self.test_instance_indexes])

    # get training and validation data chunks, given the input indexes
    def get_trainval_data(self, trainval_idx):
        """get training and validation data chunks, given the input indexes"""
        # labels
        train_labels, val_labels = self.prepare_labels(trainval_idx)
        # data
        if self.num_train != self.num_train_labels:
            trainval_idx = self.expand_index_to_sequence(trainval_idx)
        train_data, val_data = [self.process_input(data) if len(data) > 0 else np.empty((0,)) for data in
                                [self.train[idx] if len(idx) > 0 else [] for idx in trainval_idx]]
        val_datalabels = (val_data, val_labels) if val_data.size > 0 else None
        return train_data, train_labels, val_datalabels

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
            if exists(trainval_serialization_file) and self.predictions_loading_allowed:
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
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
            else:
                splitter = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        if self.do_validate_portion:
            # check if such data exists
            if exists(trainval_serialization_file) and self.predictions_loading_allowed:
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
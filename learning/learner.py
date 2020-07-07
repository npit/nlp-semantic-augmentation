from copy import deepcopy
from os import makedirs
from os.path import dirname, exists, join

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from bundle.bundle import Bundle
from bundle.datatypes import Vectors
from component.component import Component
from defs import datatypes, roles
from learning.evaluator import Evaluator
from learning.sampling import Sampler
from learning.validation import ValidationSetting
from utils import error, info, read_pickled, tictoc, write_pickled


"""
Abstract class representing a learning model
"""


class Learner(Component):

    component_name = "learner"
    name = "learner"
    save_dir = "models"
    folds = None
    fold_index = 0
    evaluator = None
    sequence_length = None
    input_aggregation = None
    train, test = None, None

    test_instance_indexes = None
    validation = None

    allow_model_loading = None
    allow_prediction_loading = None

    train_embedding = None

    def __init__(self, consumes=None):
        """Generic learning constructor
        """
        # initialize evaluation
        consumes = Vectors.name if consumes is None else consumes
        Component.__init__(self, consumes=consumes)

    # input preproc
    def process_input(self, data):
        return data

    def count_samples(self):
        self.num_train, self.num_test = map(len, [self.train_index, self.test_index])

    def read_config_variables(self):
        """Shortcut function for readding a load of config variables"""
        self.allow_prediction_loading = self.config.misc.allow_prediction_loading
        self.allow_model_loading = self.config.misc.allow_model_loading

        self.sequence_length = self.config.sequence_length

        self.results_folder = self.config.folders.results
        self.models_folder = join(self.results_folder, "models")

        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.early_stopping_patience = self.config.train.early_stopping_patience
        self.folds = self.config.train.folds
        self.validation_portion = self.config.train.validation_portion
        self.do_folds = self.folds and self.folds > 1
        self.do_validate_portion = self.validation_portion is not None and self.validation_portion > 0.0
        self.validation_exists = (self.do_folds or self.do_validate_portion)

        self.seed = self.config.misc.seed

        self.sampling_method, self.sampling_ratios = self.config.train.sampling_method, self.config.train.sampling_ratios
        self.do_sampling = self.sampling_method is not None

    def check_sanity(self):
        """Sanity checks"""
        # check data for nans
        if np.size(np.where(np.isnan(self.embeddings))[0]) > 0:
            error("NaNs exist in data:{}".format(np.where(np.isnan(self.embeddings))))
        # validation configuration
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(
                self.folds, self.validation_portion))
        if not (self.validation_exists or self.test_data_available()):
            error("No test data or cross/portion-validation setting specified.")
        self.evaluator.check_sanity()

    def configure_validation_setting(self):
        """Initialize validation setting"""
        self.validation = ValidationSetting(self.folds, self.validation_portion, self.test_data_available())
        self.validation.assign_data(self.embeddings, train_index=self.train_index, test_index=self.test_index)

    def configure_sampling(self):
        """No label-agnostic sampling"""
        # No label-agnostic sampling available
        pass

    def attach_evaluator(self):
        """Evaluator instantiation"""
        self.evaluator = Evaluator(self.config, self.embeddings, self.train_index, self.validation.use_for_testing)

    def make(self):
        # get handy variables
        self.read_config_variables()
        np.random.seed(self.seed)
        self.count_samples()

        info("Learner data: embeddings: {} train idxs: {} test idxs: {}".format(
            self.embeddings.shape, len(self.train_index), len(self.test_index)))

        info("Created learning: {}".format(self))

    def get_model_instance_name(self):
        """Get a model instance identifier from all model instances in the experiment"""
        model_name = self.name
        if self.do_folds:
            model_name += "_fold_" + str(self.fold_index)
        elif self.do_validate_portion:
            model_name += "_valportion_" + str(self.fold_index)
        return model_name

    def get_existing_predictions(self):
        path = self.validation.modify_suffix(
            join(self.results_folder, "{}".format(
                self.name))) + ".predictions.pkl"
        return read_pickled(path) if exists(path) else (None, None)

    def get_existing_trainval_indexes(self):
        """Check if the current training run is already completed."""
        trainval_file = self.get_trainval_serialization_file()
        if exists(trainval_file):
            info("Training {} with input data: {} samples on LOADED existing {}" .format(self.name, self.num_train, self.validation))
            idx = read_pickled(trainval_file)
            self.validation.check_indexes(idx)
            max_idx = max([np.max(x) for tup in idx for x in tup])
            if max_idx >= self.num_train:
                error(
                    "Mismatch between max instances in training data ({}) and loaded max index ({})."
                    .format(self.num_train, max_idx))

    def get_existing_model_path(self):
        path = self.get_current_model_path()
        return path if exists(path) else None

    def test_data_available(self):
        return len(self.test_index) > 0

    # function to retrieve training data as per the existing configuration
    def get_trainval_indexes(self):
        """Retrieve training/validation instance indexes

        :returns: 
        :rtype: 

        """
        trainval_idx = None
        # check whether they can be loaded
        if self.allow_model_loading:
            ret = self.get_existing_model_path()
            if ret:
                trainval_idx, self.existing_model_paths = ret
        if not trainval_idx:
            if not self.validation_exists:
                # return all training indexes, no validation
                return [(np.arange(len(self.train_embedding_index)), np.arange(0))]
            # compute the indexes
            trainval_idx = self.compute_trainval_indexes()
            trainval_serialization_file = self.get_trainval_serialization_file()

            # do sampling processing
            if self.do_sampling:
                smpl = Sampler()
                trainval_idx = smpl.sample()

            # save the splits
            makedirs(dirname(trainval_serialization_file), exist_ok=True)
            write_pickled(trainval_serialization_file, trainval_idx)

        # handle indexes for multi-instance data
        if self.sequence_length > 1:
            self.validation.set_trainval_label_index(deepcopy(trainval_idx))
            # trainval_idx = self.expand_index_to_sequence(trainval_idx)
        return trainval_idx

    def acquire_trained_model(self):
        """Trains the learning model or load an existing instance from a persisted file."""
        with tictoc("Training run [{}] on {} training and {} val data.".format(self.validation, self.num_train, len(self.val_index) if self.val_index is not None else "[none]")):
            model = None
            # check if a trained model already exists
            if self.allow_model_loading:
                model = self.load_model()
            if not model:
                model = self.train_model()
                # create directories
                makedirs(self.models_folder, exist_ok=True)
                self.save_model(model)
            else:
                info("Skipping training due to existing model successfully loaded.")
        return model

    def assign_current_run_data(self, iteration_index, trainval):
        """Sets data for the current run"""
        self.train_index, self.val_index, self.test_index, self.test_instance_indexes = self.validation.get_run_data(iteration_index, trainval)
        self.num_train, self.num_val, self.num_test = [len(x) for x in (self.train_index, self.val_index, self.test_index)]

    # perfrom a train-test loop
    def do_traintest(self):
        with tictoc("Entire learning run", do_print=self.do_folds, announce=False):

            # get trainval data
            train_val_idxs = self.get_trainval_indexes()

            # keep track of models' test performances and paths wrt selected metrics
            model_paths = []

            # iterate over required runs as per the validation setting
            for iteration_index, trainval in enumerate(train_val_idxs):
                self.assign_current_run_data(iteration_index, trainval)

                # for evaluation, pass all information of the current (maybe cross-validated) run testing
                self.evaluator.update_reference_data(train_index=self.train_index, test_index=self.test_index)

                # check if the run is completed already and load existing results, if allowed
                model, predictions = None, None
                if self.allow_prediction_loading:
                    predictions, test_instance_indexes = self.load_existing_predictions()

                # train the model
                if predictions is None:
                    model = self.acquire_trained_model()
                else:
                    info("Skipping training due to existing predictions successfully loaded.")

                # test and evaluate the model
                with tictoc("Testing run [{}] on {} test data.".format(self.validation.descr, self.num_test)):
                    self.do_test_evaluate(model, predictions)
                    model_paths.append(self.get_current_model_path())

                self.conclude_validation_iteration()

            # end of entire train-test loop
            self.conclude_traintest()
            # report results across entire training
            self.evaluator.report_overall_results(self.validation.descr, self.results_folder)

    # evaluate a model on the test set
    def do_test_evaluate(self, model, predictions=None):
        if predictions is None:
            # evaluate the model
            error("No test data supplied!", len(self.test_index) == 0)
            predictions = self.test_model(model)
        # get performances
        self.evaluator.evaluate_learning_run(predictions, self.test_instance_indexes)
        if self.do_folds and self.config.print.folds:
            self.evaluator.print_run_performance(self.validation.descr, self.validation.current_fold)
        # write fold predictions
        predictions_file = self.validation.modify_suffix(join(self.results_folder, "{}".format(self.name))) + ".predictions.pkl"
        write_pickled(predictions_file, [predictions, self.test_instance_indexes])

    def conclude_traintest(self):
        """Perform concuding actions for the entire traintest loop"""
        pass

    def conclude_validation_iteration(self):
        """Perform concluding actions for a single validation loop"""
        # wrap up the current validation iteration
        self.validation.conclude_iteration()

    def get_current_model_path(self):
        return self.validation.modify_suffix(
            join(self.results_folder, "models", "{}".format(self.name))) + ".model"

    def get_trainval_serialization_file(self):
        sampling_suffix = "{}.trainvalidx.pkl".format(
            "" if not self.do_sampling else "{}_{}".
            format(self.sampling_method, "_".
                   join(map(str, self.sampling_ratios))))
        return self.validation.modify_suffix(
            join(self.results_folder, "{}".format(
                self.name))) + sampling_suffix

    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):
        """Compute training/validation indexes

        :returns: The training/validation indexes
        :rtype: list of ndarray tuples

        """

        if self.do_folds:
            info("Splitting {} with input data: {} samples, on {} folds"
                .format(self.name, self.num_train, self.folds))
            splitter = KFold(self.folds, shuffle=True, random_state=self.seed)

        if self.do_validate_portion:
            info("Splitting {} with input data: {} samples on a {} validation portion"
                .format(self.name, self.num_train, self.validation_portion))
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=self.validation_portion,
                random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(splitter.split(np.zeros(self.num_train)))
        return splits

    def is_supervised(self):
        error("Attempted to access abstract learner is_supervised() method")
        return None

    def save_model(self, model):
        path = self.get_current_model_path()
        info("Saving model to {}".format(path))
        write_pickled(path, model)

    def load_model(self):
        path = self.get_current_model_path()
        if not path or not exists(path):
            return None
        info("Loading existing learning model from {}".format(path))
        return read_pickled(self.get_current_model_path())

    # region: component functions
    def run(self):
        self.process_component_inputs()
        self.make()
        self.configure_validation_setting()
        self.attach_evaluator()
        self.configure_sampling()
        self.check_sanity()
        self.do_traintest()
        self.outputs.set_vectors(Vectors(vecs=self.evaluator.predictions["run"]))

    def acquire_embedding_information(self):
        """Get embedding and embedding information"""
        # get data
        error("Vanilla learner needs at least two-input bundle input list.", len(self.inputs) <= 1)
        error(f"{self.name} {self.component_name} needs vector information.", not self.inputs.has_vectors())
        error(f"{self.name} {self.component_name} needs vector information.", not self.inputs.has_indices())

        # get vectors and their indices
        vectors_bundle = self.inputs.get_vectors(full_search=True, enforce_single=True)

        self.embeddings = vectors_bundle.get_vectors().instances
        self.train_embedding_index = vectors_bundle.get_indices(role=roles.train, enforce_single=True)
        # get_train self.inputs.get_indices(single=True, role=roles.train)

        if vectors_bundle.get_indices().has_role(roles.test):
            self.test_embedding_index = vectors_bundle.get_indices(role=roles.test, enforce_single=True)
        else:
            self.test_embedding_index = np.ndarray((), np.float32)

    def process_component_inputs(self):
        """Component processing for input indexes and vectors"""
        self.acquire_embedding_information()
        # initialize current meta-indexes to data
        self.train_index = np.arange(len(self.train_embedding_index))
        self.test_index = np.arange(len(self.test_embedding_index))

    def load_existing_predictions(self):
        """Loader function for existing, already computed predictions"""
        # get predictions and instance indexes they correspond to
        existing_predictions, existing_instance_indexes = self.get_existing_predictions()
        if existing_predictions is not None:
            info("Loaded existing predictions.")
        if not np.all(np.equal(existing_instance_indexes, self.test_instance_indexes)):
            error("Different instance indexes loaded than the ones generated.")
        return existing_predictions, existing_instance_indexes

    def get_data_from_index(self, index, embeddings):
        """Get data index from the embedding matrix"""
        if np.squeeze(index).ndim > 1:
            if self.input_aggregation is None and self.sequence_length < 2:
                # if we have multi-element index, there has to be an aggregation method defined for the learner.
                error("Learner [{}] has no defined aggregation and is not sequence-capable, but the input index has shape {}".format(self.name, index.shape))
        return embeddings[index] if len(index) > 0 else None

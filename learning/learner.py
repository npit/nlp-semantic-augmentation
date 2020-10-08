from copy import deepcopy
from os import makedirs
from os.path import dirname, exists, join, basename, abspath, isabs

import numpy as np

from bundle.bundle import DataPool
from bundle.datatypes import *
from bundle.datausages import *
from serializable import Serializable
from component.component import Component
from defs import datatypes, roles
from learning.evaluator import Evaluator
from learning.sampling import Sampler
from learning.validation.validation import ValidationSetting, get_info_string
from utils import error, info, read_pickled, tictoc, write_pickled, warning


"""
Abstract class representing a learning model
"""


class Learner(Serializable):

    component_name = "learner"
    name = "learner"
    save_dir = "models"
    folds = None
    fold_index = 0
    evaluator = None

    test_instance_indexes = None
    validation = None

    allow_model_loading = None
    allow_prediction_loading = None

    train_embedding = None

    # store information pertaining to the learning run(s) executed
    predictions = []
    prediction_tags = []
    prediction_roles = []
    prediction_indexes = []
    models = []

    def __init__(self, consumes=None):
        """Generic learning constructor
        """
        # initialize evaluation
        Serializable.__init__(self, "")

    # input preproc
    def process_input(self, data):
        return data

    def count_samples(self):
        self.num_train, self.num_test = map(len, [self.train_index, self.test_index])

    def read_config_variables(self):
        """Shortcut function for readding a load of config variables"""
        self.allow_prediction_loading = self.config.allow_prediction_loading
        self.allow_model_loading = self.config.misc.allow_model_loading

        self.explicit_model_path = self.config.explicit_model_path

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
        self.save_interval = self.config.save_interval

    def check_sanity(self):
        """Sanity checks"""
        # check data for nans
        if np.size(np.where(np.isnan(self.embeddings))[0]) > 0:
            error("NaNs exist in data:{}".format(np.where(np.isnan(self.embeddings))))
        # validation configuration
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(
                self.folds, self.validation_portion))
        # self.evaluator.check_sanity()

    def configure_validation_setting(self):
        """Initialize validation setting"""
        self.validation = ValidationSetting(self.config, self.train_embedding_index, self.test_embedding_index, folds=self.folds, portion=self.validation_portion, seed=self.seed)

    def configure_sampling(self):
        """No label-agnostic sampling"""
        # No label-agnostic sampling available
        pass

    def get_all_preprocessed(self):
        return (self.predictions, self.prediction_tags, self.prediction_indexes)

    def make(self):
        # get handy variables
        self.read_config_variables()
        np.random.seed(self.seed)
        self.count_samples()

        info("Learner data: embeddings: {} train idxs: {} test idxs: {}".format(
            self.embeddings.shape, len(self.train_index), len(self.test_index)))

        info("Created learner: {}".format(self))

    def get_model_instance_name(self):
        """Get a model instance identifier from all model instances in the experiment"""
        model_name = self.name
        if self.config.train.folds:
            model_name += "_fold_" + str(self.fold_index)
        elif self.config.train.validation_portion:
            model_name += "_valportion_" + str(self.fold_index)
        return model_name

    def get_predictions_file(self, tag="test"):
        if tag:
            tag += "."
        return join(self.get_results_folder(),  self.get_model_filename() + "." + tag + "predictions.pkl")

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


    # function to retrieve training data as per the existing configuration
    def configure_trainval_indexes(self):
        """Retrieve training/validation instance indexes

        :returns: 
        :rtype: 

        """
        trainval_idx = self.validation.get_trainval_indexes()

        # get train/val splits
        trainval_serialization_file = join(self.config.folders.run, self.get_trainval_serialization_file())

        # do sampling 
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
                self.save_model()
            else:
                info("Skipping training due to existing model successfully loaded.")
        return model

    # def assign_current_run_data(self, iteration_index, trainval):
    #     """Sets data for the current run"""
    #     self.train_index, self.val_index = self.validation.get_
    #     self.train_index, self.val_index, self.test_instance_indexes = self.validation.get_run_data(iteration_index, trainval)
    #     self.num_train, self.num_val, self.num_test = [len(x) for x in (self.train_index, self.val_index, self.test_index)]

    # perfrom a train-test loop
    def do_traintest(self):
        with tictoc("Learning run", do_print=self.do_folds, announce=False):

            # get training - validation instance indexes for building the model
            self.configure_trainval_indexes()

            # iterate over required runs (e.g. portion split or folds)
            # # as per the validation setting
            for iteration_index, trainval in enumerate(self.validation.get_trainval_indexes()):
                # set the train/val data indexes
                self.train_index, self.val_index = trainval
                
                # self.prediction_indexes.append(self.validation.get_prediction_indexes())

                # train and keep track of the model
                model = self.acquire_trained_model()
                self.models.append(model)

                # self.conclude_validation_iteration()

            # end of entire train-test loop
            self.conclude_traintest()


    def conclude_traintest(self):
        """Perform concuding actions for the entire traintest loop"""
        pass

    def get_current_model_path(self):
        if self.config.explicit_model_path is not None:
            return self.config.explicit_model_path
        folder = join(self.get_results_folder(), "models")
        return join(folder, self.get_model_filename())

    def get_model_filename(self):
        """Retrieve model filename"""
        return self.name + get_info_string(self.config) + ".model"

    def get_results_folder(self):
        """Return model folder"""
        run_folder = self.config.folders.results
        if not isabs(run_folder):
            run_folder = join(abspath(run_folder))
        return run_folder

    def get_trainval_serialization_file(self):
        name = "trainvalidx.pkl"
        if self.do_sampling:
            name = self.sampling_method + "_" + "_". join(map(str, self.sampling_ratios)) + "." + name
        if self.validation:
            name += self.validation.get_info_string() 
        return name

    # region: component functions
    def load_outputs_from_disk(self):
        self.set_serialization_params()
        self.add_serialization_source(self.get_predictions_file(), reader=read_pickled, handler=lambda x: x)
        return self.load_existing_predictions()

    def load_existing_predictions(self):
        """Loader function for existing, already computed predictions. Checks for label matching."""
        # get predictions and instance indexes they correspond to
        try:
            self.predictions, self.prediction_indexes = read_pickled(self.get_predictions_file())
        except FileNotFoundError:
            return False
        # also check labels
        return True







    def build_model_from_inputs(self):
        self.make()
        self.configure_validation_setting()
        if self.validation_exists and not len(self.test_embedding_index) > 0:
            self.validation.reserve_validation_for_testing()
        # self.attach_evaluator()
        self.configure_sampling()
        self.check_sanity()
        self.serialization_path_preprocessed = join(self.results_folder, "data")
        self.do_traintest()

    def produce_outputs(self):
        # apply the learning model on the input data
        # produce pairing with ground truth for future evaluation
        # training data
        if self.validation is not None:
            train_indexes = self.validation.get_train_indexes()
            test_indexes = self.validation.get_test_indexes()
        else:
            train_indexes = [self.train_embedding_index]
            test_indexes = [self.test_embedding_index]

        for idx_group, role in zip([train_indexes, test_indexes], [defs.roles.train, defs.roles.test]):
            if len(idx_group) > 1:
                info(f"Applying {self.name} model on {len(idx_group)} {role} data")
            for i, idxs in enumerate(idx_group):
                if len(idxs) == 0:
                    info(f"Skipping model application on{role} data since it has no instances.")
                    continue
                info(f"Applying {self.name} model on {len(idxs)} {role} data")
                tag = role + f"{i+1}" if len(idx_group)>1 else role
                self.apply_model(index=idxs, tag=tag)
                self.prediction_roles.append(role)

        # with tictoc(f"Applying the {self.name} on the training data."):
        #     for i, train_idx in enumerate(train_indexes):
        #         tag = f"train{i+1}" if len(test_indexes) > 0 else "train"
        #         self.apply_model(index=train_idx, tag=tag)
        # # test data
        # with tictoc(f"Applying the {self.name} on the test data."):
        #     for i, test_idx in enumerate(test_indexes):
        #         tag = f"test{i+1}" if len(test_indexes) > 0 else "test"
        #         self.apply_model(index=test_idx, tag=tag)

        # self.test_index = self.test_embedding_index
        # if self.test_index.size > 0:
        #     self.num_test = len(self.test_embedding_index)
        #     with tictoc(f"Testing run on {self.num_test} test data."):
        #         self.do_test_evaluate(self.model)

    def get_model(self):
        return self.model
    # evaluate a model on the test set
    def apply_model(self, model=None, index=None, tag="test"):
        """Evaluate the model on the current test indexes"""
        if model is None:
            model = self.get_model()
        if index is not None:
            self.test_index = index
        if len(self.test_index) == 0:
            warning(f"Attempted to apply {self.name} model on empty indexes!")
            return
        predictions = self.test_model(model)
        # write the predictions and relevant idxs (i.e. if testing on validation)
        # predictions_file = self.validation.modify_suffix(join(self.results_folder, "{}".format(self.name))) + ".predictions.pkl"
        predictions_file = self.get_predictions_file(tag) 
        write_pickled(predictions_file, [predictions, self.test_instance_indexes])
        self.predictions.append(predictions)
        self.prediction_tags.append(tag)
        self.prediction_indexes.append(self.test_index)

    def load_model_from_disk(self):
        """Load the component's model from disk"""
        return self.load_model()

    def acquire_embedding_information(self):
        """Get embedding and embedding information"""
        # get data
        vectors = self.data_pool.request_data(Numeric, Indices.name, self.name)
        self.embeddings = vectors.data.instances
        self.indices = vectors.get_usage(Indices)
        self.train_embedding_index, self.test_embedding_index = self.indices.get_train_test()
        # self.test_embedding_index = self.indices.get_role_instances(roles.train)
        # if self.indices.has_role(roles.test):
        #     self.test_embedding_index = self.indices.get_role_instances(roles.test)
        # else:
        #     self.test_embedding_index = np.ndarray((0,), np.float32)


        # self.embeddings = vectors_bundle.get_vectors().instances
        # self.train_embedding_index = vectors_bundle.get_indices(role=roles.train, enforce_single=True)
        # # get_train self.inputs.get_indices(single=True, role=roles.train)

        # if vectors_bundle.get_indices().has_role(roles.test):
        #     self.test_embedding_index = vectors_bundle.get_indices(role=roles.test, enforce_single=True)
        # else:
        #     self.test_embedding_index = np.ndarray((), np.float32)

    def get_component_inputs(self):
        """Component processing for input indexes and vectors"""
        self.acquire_embedding_information()
        # initialize current meta-indexes to data
        self.train_index = np.arange(len(self.train_embedding_index))
        self.test_index = np.arange(len(self.test_embedding_index))


    def get_data_from_index(self, index, embeddings):
        """Get data index from the embedding matrix"""
        if np.squeeze(index).ndim > 1:
            if self.input_aggregation is None and self.sequence_length < 2:
                # if we have multi-element index, there has to be an aggregation method defined for the learner.
                error("Learner [{}] has no defined aggregation and is not sequence-capable, but the input index has shape {}".format(self.name, index.shape))
        return embeddings[index] if len(index) > 0 else None

    def set_component_outputs(self):
        """Set the output data of the clusterer"""
        # predictions to output
        # predictions
        pred = Numeric(self.predictions)
        pred = DataPack(pred, Predictions(self.prediction_indexes, roles=self.prediction_roles))
        self.data_pool.add_data_packs([pred], self.name)


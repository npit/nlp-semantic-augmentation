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
from learning.validation.validation import ValidationSetting, get_info_string, load_trainval
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


    train_embedding = None
    model_index = None

    def __init__(self, consumes=None):
        """Generic learning constructor
        """
        # initialize evaluation
        Serializable.__init__(self, "")
        self.models = []
        self.predictions = []
        self.prediction_roles = []
        self.prediction_model_indexes = []

    # input preproc
    def process_input(self, data):
        return data

    def read_config_variables(self):
        """Shortcut function for readding a load of config variables"""

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
        self.validation = ValidationSetting(self.config, self.train_embedding_index,
                                            self.test_embedding_index,
                                            folds=self.folds,
                                            portion=self.validation_portion,
                                            seed=self.seed)

    def configure_sampling(self):
        """No label-agnostic sampling"""
        # No label-agnostic sampling available
        pass

    def get_all_preprocessed(self):
        return (self.predictions, self.output_usage.to_json())

    def make(self):
        # get handy variables
        self.read_config_variables()
        np.random.seed(self.seed)

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

    def get_predictions_file(self, model_index=None, tag="test"):
        if tag:
            tag += "."
        return join(self.get_results_folder(),  self.get_model_filename(model_index) + "." + tag + "predictions.pkl")

    def get_existing_model_path(self):
        path = self.get_current_model_path()
        return path if exists(path) else None

    def get_model_path(self):
        return self.get_current_model_path()

    # function to retrieve training data as per the existing configuration
    def configure_trainval_indexes(self):
        """Retrieve training/validation instance indexes
        """
        trainval_idx = self.validation.get_trainval_indexes()

        # do sampling 
        if self.do_sampling:
            smpl = Sampler()
            trainval_idx = smpl.sample()

        # handle indexes for multi-instance data
        if self.sequence_length > 1:
            self.validation.set_trainval_label_index(deepcopy(trainval_idx))
            # trainval_idx = self.expand_index_to_sequence(trainval_idx)

    def acquire_trained_model(self):
        """Trains the learning model or load an existing instance from a persisted file."""
        with tictoc("Training run [{}] - {} on {} training and {} val data.".format(get_info_string(self.config), self.model_index, len(self.train_index), len(self.val_index) if self.val_index is not None else "[none]")):
            model = None
            if not model:
                model = self.train_model()
                # create directories
                makedirs(self.models_folder, exist_ok=True)
            else:
                info("Skipping training due to existing model successfully loaded.")
        return model

    # def assign_current_run_data(self, iteration_index, trainval):
    #     """Sets data for the current run"""
    #     self.train_index, self.val_index = self.validation.get_
    #     self.train_index, self.val_index, self.test_instance_indexes = self.validation.get_run_data(iteration_index, trainval)
    #     self.num_train, self.num_val, self.num_test = [len(x) for x in (self.train_index, self.val_index, self.test_index)]

    # perfrom a train-test loop
    def execute_training(self):
        with tictoc("Training run", do_print=self.do_folds, announce=False):

            # get training - validation instance indexes for building the model
            self.configure_trainval_indexes()

            # iterate over required runs (e.g. portion split or folds)
            # # as per the validation setting
            for iteration_index, trainval in enumerate(self.validation.get_trainval_indexes()):
                # set the train/val data indexes
                self.train_index, self.val_index = trainval

                # train and keep track of the model
                self.model_index = iteration_index
                model = self.acquire_trained_model()
                self.append_model_instance(model)

                # self.conclude_validation_iteration()

    def append_model_instance(self, model):
        self.models.append(model)

    def get_current_model_path(self):
        if self.config.explicit_model_path is not None:
            return self.config.explicit_model_path
        folder = join(self.get_results_folder(), "models")
        return join(folder, self.get_model_filename())

    def get_model_filename(self, model_index=None):
        """Retrieve model filename"""
        if model_index is None:
            model_index = self.model_index
        model_id = "" if model_index is None else f".model_{model_index}"
        return self.name + get_info_string(self.config) + model_id  + ".model"

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
        self.add_serialization_source(self.get_predictions_file("total"), reader=read_pickled, handler=lambda x: x)
        return self.load_existing_predictions()

    def load_existing_predictions(self):
        """Loader function for existing, already computed predictions. Checks for label matching."""
        # get predictions and instance indexes they correspond to
        try:
            self.predictions, pred_instances, pred_tags = read_pickled(self.get_predictions_file("total"))
            self.output_usage = Predictions(pred_instances, pred_tags)
        except FileNotFoundError:
            return False
        # also check labels
        return True

    def build_model_from_inputs(self):
        self.make()
        self.configure_validation_setting()

        if self.validation_exists and not len(self.test_embedding_index) > 0:
            self.validation.reserve_validation_for_testing()
        output_file = self.get_current_model_path() + ".trainval_idx"
        self.validation.write_trainval(output_file)
        # self.attach_evaluator()
        self.configure_sampling()
        self.check_sanity()
        self.serialization_path_preprocessed = join(self.results_folder, "data")
        self.execute_training()

    def produce_outputs(self):
        # apply the learning model on the input data
        # produce pairing with ground truth for future evaluation
        # training data
        self.configure_model_after_inputs()
        self.predictions = None
        self.prediction_model_indexes = []
        self.prediction_roles = []

        if self.validation is not None:
            train_indexes = self.validation.get_train_indexes()
            test_indexes = self.validation.get_test_indexes()
        else:
            train_indexes = [self.train_embedding_index]
            test_indexes = [self.test_embedding_index]

        # loop over the available models / data batches
        num_models = len(self.models)
        if num_models > 1:
            if len(train_indexes) == 1:
                # single indexes, multiple models: duplicate
                train_indexes *= num_models
                test_indexes *= num_models

        # make the output usage object
        self.output_usage = None
        for model_index, model in enumerate(self.models):
            self.model_index = model_index
            train, test = train_indexes[model_index], test_indexes[model_index]
            for (data, role) in zip([train, test], [defs.roles.train, defs.roles.test]):
                if len(data) > 0:
                    info(f"Evaluating model {model_index + 1}/{num_models} on {len(data)} {role} data")
                    # tag = f"model_{self.model_index}_{role}"
                    self.apply_model(model=model, index=data, tag=role)
        # no predictions in the output
        if self.predictions is None:
            self.predictions = np.empty(0)
            self.output_usage = Predictions(np.empty(0, dtype=np.int64), "dummy")

    def get_model(self):
        return self.model

    # evaluate a model on the test set
    def apply_model(self, model=None, index=None, tag="test"):
        """Evaluate the model on the current test indexes"""
        if model is None:
            model = self.models[self.model_index]
        if index is not None:
            self.test_index = index
        if len(self.test_index) == 0:
            predictions = np.empty(0)
        else:
            # generate predictions
            predictions = self.test_model(model)

        pred_idx = np.arange(len(predictions))
        # keep track output predictions and tags
        if self.predictions is None:
            self.predictions = predictions
        else:
            pred_idx += len(self.predictions)
            self.predictions = np.append(self.predictions, predictions, axis=0)
        
        # mark the model
        model_id = f"model_{self.model_index}"
        if self.output_usage is None:
            self.output_usage = Predictions(pred_idx, model_id)
        else:
            self.output_usage.add_instance(pred_idx, model_id)

        # mark the tag
        self.output_usage.add_instance(pred_idx, tag)
        # mark the correspondence to the input instances
        self.output_usage.add_instance(self.test_index, f"{model_id}_{tag}_{defs.roles.inputs}")

    def save_outputs(self):
        """Save produced predictions"""
        predictions_file = self.get_predictions_file("total") 
        write_pickled(predictions_file, [self.predictions, self.output_usage.instances, self.output_usage.tags])

    def load_model_from_disk(self):
        """Load the component's model from disk"""
        model_loaded = self.load_model()
        if model_loaded:
            self.append_model_instance(self.get_model())
        return model_loaded

    def acquire_embedding_information(self):
        """Get embedding and embedding information"""
        # get data
        vectors = self.data_pool.request_data(Numeric, Indices.name, self.name)
        self.embeddings = vectors.data.instances
        self.indices = vectors.get_usage(Indices)
        self.train_embedding_index, self.test_embedding_index = self.indices.get_train_test()

    def get_component_inputs(self):
        """Component processing for input indexes and vectors"""
        self.acquire_embedding_information()
        # initialize current meta-indexes to data
        self.train_index = np.arange(len(self.train_embedding_index))
        self.test_index = np.arange(len(self.test_embedding_index))

    def configure_model_after_inputs(self):
        pass

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
        dp = self.make_predictions_datapack()
        self.data_pool.add_data_packs([dp], self.name)

    def make_predictions_datapack(self):
        pred = DataPack(Numeric(self.predictions), self.output_usage)
        return pred

    def load_model_wrapper(self):
        return True

    def save_model_wrapper(self):
        pass        

    def save_model(self):
        for m in range(len(self.models)):
            self.model_index = m
            path = self.get_current_model_path()
            super().save_model(path=path, model=self.models[m])
        self.save_model_wrapper()


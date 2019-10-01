from copy import deepcopy
from os import makedirs
from os.path import basename, dirname, exists, join

import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     StratifiedShuffleSplit)

import defs
from augmentation.augmentation import LabelledDataAugmentation
from bundle.bundle import Bundle, BundleList
from bundle.datatypes import Labels, Vectors
from component.component import Component
from learning.evaluator import Evaluator
from utils import (count_label_occurences, error, info, is_multilabel, one_hot,
                   read_pickled, tictoc, warning, write_pickled)


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
    train, test, train_labels, test_labels = None, None, None, None

    test_instance_indexes = None
    validation = None

    allow_model_loading = None
    allow_prediction_loading = None

    train_embedding = None

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
            map(len, [self.train_index, self.test_index, self.train_labels, self.test_labels])

    def make(self):
        self.verbosity = 1 if self.config.print.training_progress else 0
        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(
                zero_samples_idx))
        # nan checks
        nans, _ = np.where(np.isnan(self.embeddings))
        if np.size(nans) != 0:
            error("NaNs in training data:{}".format(nans))
        nans, _ = np.where(np.isnan(self.embeddings))
        if np.size(nans) != 0:
            error("NaNs in test data:{}".format(nans))

        # get many handy variables
        self.do_multilabel = is_multilabel(self.train_labels)
        label_counts = count_label_occurences(self.train_labels)
        self.num_labels = len(label_counts)
        self.count_samples()
        self.input_dim = self.embeddings.shape[-1]
        self.allow_prediction_loading = self.config.misc.allow_prediction_loading
        self.allow_model_loading = self.config.misc.allow_model_loading
        self.sequence_length = self.config.learner.sequence_length
        self.train_embedding = self.config.learner.train_embedding
        self.results_folder = self.config.folders.results
        self.models_folder = join(self.results_folder, "models")
        self.epochs = self.config.train.epochs
        self.folds = self.config.train.folds
        self.validation_portion = self.config.train.validation_portion
        self.do_folds = self.folds and self.folds > 1
        self.do_validate_portion = self.validation_portion is not None and self.validation_portion > 0.0
        self.validation_exists = (self.do_folds or self.do_validate_portion)
        self.use_validation_for_training = self.validation_exists and self.test_data_available(
        )
        self.early_stopping_patience = self.config.train.early_stopping_patience

        self.validation = Learner.ValidatonSetting(self.folds,
                                                   self.validation_portion,
                                                   self.test_data_available(),
                                                   self.do_multilabel)
        self.validation.assign_data(self.embeddings, self.train_index, self.train_labels, self.test_labels, self.test_index)

        # sampling
        self.sampling_method, self.sampling_ratios = self.config.train.sampling_method, self.config.train.sampling_ratios
        self.do_sampling = self.sampling_method is not None
        if self.do_sampling:
            if type(self.sampling_ratios[0]) is not list:
                self.sampling_ratios = [self.sampling_ratios]
            freqs = count_label_occurences(
                [x for y in self.sampling_ratios for x in y[:2]])
            max_label_constraint_participation = max(freqs, key=lambda x: x[1])
            error(
                "Sampling should be applied on binary classification or constraining ratio should not be overlapping",
                self.num_labels > 2
                and max_label_constraint_participation[1] > 1)

        self.seed = self.config.misc.seed
        np.random.seed(self.seed)

        self.batch_size = self.config.train.batch_size
        info("Learner data/labels: embeddings: {} train idxs: {} test idxs: {}".format(
            self.embeddings.shape, len(self.train_index), len(self.test_index)))

        # sanity checks
        if self.do_folds and self.do_validate_portion:
            error("Specified both folds {} and validation portion {}.".format(
                self.folds, self.validation_portion))
        if not (self.validation_exists or self.test_data_available()):
            error(
                "No test data or cross/portion-validation setting specified.")

        # configure and sanity-check evaluator
        if self.validation_exists and not self.use_validation_for_training:
            # calculate the majority label from the training data -- label counts already computed
            self.evaluator.majority_label = label_counts[0][0]
            info("Majority label: {}".format(self.evaluator.majority_label))
            self.evaluator.configure(self.train_labels, self.num_labels,
                                     self.do_multilabel,
                                     self.use_validation_for_training,
                                     self.validation_exists)
            self.evaluator.compute_label_distribution()
        else:
            # count label distribution from majority
            self.evaluator.configure(self.test_labels, self.num_labels,
                                     self.do_multilabel,
                                     self.use_validation_for_training,
                                     self.validation_exists)
            self.evaluator.majority_label = count_label_occurences(
                self.test_labels)[0][0]

        error("Input none dimension.", self.input_dim is None)
        info("Created learning: {}".format(self))

    def get_existing_predictions(self):
        path = self.validation.modify_suffix(
            join(self.results_folder, "{}".format(
                self.name))) + ".predictions.pickle"
        return read_pickled(path) if exists(path) else (None, None)

    def get_existing_trainval_indexes(self):
        """Check if the current training run is already completed."""
        trainval_file = self.get_trainval_serialization_file()
        if exists(trainval_file):
            info(
                "Training {} with input data: {} samples, {} labels, on LOADED existing {}"
                .format(self.name, self.num_train, self.num_train_labels,
                        self.validation))
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
        trainval_idx = None
        # get training / validation indexes
        if self.allow_model_loading:
            ret = self.get_existing_model_path()
            if ret:
                trainval_idx, self.existing_model_paths = ret
        if not trainval_idx:
            trainval_idx = self.compute_trainval_indexes()

        # handle indexes for multi-instance data
        # if self.num_train != self.num_train_labels:
        if self.sequence_length > 1:
            self.validation.set_trainval_label_index(deepcopy(trainval_idx))
            # trainval_idx = self.expand_index_to_sequence(trainval_idx)
        return trainval_idx

    # perfrom a train-test loop
    def do_traintest(self):
        with tictoc("Entire learning run",
                    do_print=self.do_folds,
                    announce=False):

            # get trainval data
            train_val_idxs = self.get_trainval_indexes()

            # keep track of models' test performances and paths wrt selected metrics
            model_paths = []

            # iterate over required runs as per the validation setting
            for iteration_index, trainval in enumerate(train_val_idxs):
                train_index, train_labels, val_index, val_labels, \
                test_index, test_labels, test_instance_indexes = self.validation.get_run_data(iteration_index, trainval)

                # show training data statistics
                self.evaluator.show_label_distribution(
                    count_label_occurences(train_labels),
                    "Training label distribution for validation setting: " +
                    self.validation.get_current_descr())
                if val_index is not None:
                    self.evaluator.show_label_distribution(
                        count_label_occurences(val_labels),
                        "Validation label distribution for validation setting: "
                        + self.validation.get_current_descr())

                # preprocess data and labels -- we only have indexes here.
                # assign containers
                # train_index, train_labels = train
                # test_index, test_labels = test
                # val_index, val_labels = val if val else (None, None)
                self.count_samples()

                # for cross-validation testing, gotta keep the ref. labels updated
                self.evaluator.update_reference_labels(test_labels,
                                                       train_labels)
                # check if the run is completed already and load existing results, if allowed
                model, predictions = None, None
                if self.allow_prediction_loading:
                    predictions, test_instance_indexes = self.load_existing_predictions(test_instance_indexes, test_labels)

                # train the model
                if predictions is None:
                    with tictoc( "Training run [{}] on {} training and {} val data.".format(self.validation, len(train_labels), len(val_labels) if val_labels is not None else "[none]")):
                        # check if a trained model already exists
                        if self.allow_model_loading:
                            model = self.load_model()
                        if not model:
                            model = self.train_model(train_index, self.embeddings, train_labels, val_index, val_labels)
                            # create directories
                            makedirs(self.models_folder, exist_ok=True)
                            self.save_model(model)
                        else:
                            info( "Skipping training due to existing model successfully loaded.")
                else:
                    info( "Skipping training due to existing predictions successfully loaded.")

                # test the model
                with tictoc("Testing run [{}] on {} test data.".format(
                        self.validation.descr, self.num_test_labels)):
                    self.do_test_evaluate(model, test_index, self.embeddings, test_labels,
                                          test_instance_indexes, predictions)
                    model_paths.append(self.get_current_model_path())

                if self.validation_exists and not self.use_validation_for_training:
                    self.test, self.test_labels = [], []
                    self.test_instance_indexes = None

                # wrap up validation iteration
                self.validation.conclude_iteration()

            if self.validation.use_validation_for_testing:
                # for the final evaluation, pass the entire training labels
                self.evaluator.configure(self.train_labels, self.num_labels,
                                         self.do_multilabel,
                                         self.use_validation_for_training,
                                         self.validation_exists)
                # show the overall training label distribution
                self.evaluator.show_label_distribution(
                    message="Overall training label distribution")
            else:
                # show the test label distribution
                self.evaluator.show_label_distribution()
            self.evaluator.report_overall_results(self.validation.descr,
                                                  len(self.train_index),
                                                  self.results_folder)


    # evaluate a model on the test set
    def do_test_evaluate(self,
                         model,
                         test_index,
                         embeddings,
                         test_labels,
                         test_instance_indexes,
                         predictions=None):
        if predictions is None:
            # evaluate the model
            error("No test data supplied!", len(test_index) == 0)
            predictions = self.test_model(test_index, embeddings, model)
        # get baseline performances
        self.evaluator.evaluate_learning_run(predictions, test_instance_indexes)
        if self.do_folds and self.config.print.folds:
            self.evaluator.print_run_performance(self.validation.descr, self.validation.current_fold)
        # write fold predictions
        predictions_file = self.validation.modify_suffix(join(self.results_folder, "{}".format(self.name))) + ".predictions.pickle"
        write_pickled(predictions_file, [predictions, test_instance_indexes])

    def get_current_model_path(self):
        return self.validation.modify_suffix(
            join(self.results_folder, "models", "{}".format(
                self.name))) + ".model"

    def get_trainval_serialization_file(self):
        sampling_suffix = "{}.trainvalidx.pickle".format(
            "" if not self.do_sampling else "{}_{}".
            format(self.sampling_method, "_".
                   join(map(str, self.sampling_ratios))))
        return self.validation.modify_suffix(
            join(self.results_folder, "{}".format(
                self.name))) + sampling_suffix

    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):
        if not self.validation_exists:
            return [(np.arange(self.num_train_labels), np.arange(0))]

        trainval_serialization_file = self.get_trainval_serialization_file()

        if self.do_folds:
            info(
                "Training {} with input data: {} samples, {} labels, on {} stratified folds"
                .format(self.name, self.num_train, self.num_train_labels,
                        self.folds))
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                splitter = KFold(self.folds,
                                 shuffle=True,
                                 random_state=self.seed)
            else:
                splitter = StratifiedKFold(self.folds,
                                           shuffle=True,
                                           random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        if self.do_validate_portion:
            info(
                "Splitting {} with input data: {} samples, {} labels, on a {} validation portion"
                .format(self.name, self.num_train, self.num_train_labels,
                        self.validation_portion))
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.validation_portion,
                random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(
            splitter.split(np.zeros(self.num_train_labels), self.train_labels))

        # do sampling processing
        if self.do_sampling:
            for split_index, (tr, vl) in enumerate(splits):
                aug_indexes = []
                orig_tr_size = len(tr)
                ldaug = LabelledDataAugmentation()
                if self.sampling_method == defs.sampling.oversample:
                    for (label1, label2, ratio) in self.sampling_ratios:
                        aug_indexes.append(
                            ldaug.oversample_to_ratio(self.train,
                                                      self.train_labels,
                                                      [label1, label2],
                                                      ratio,
                                                      only_indexes=True,
                                                      limit_to_indexes=tr))
                        info(
                            "Sampled via {}, to ratio {}, for labels {},{}. Modification size: {} instances."
                            .format(self.sampling_method, ratio, label1,
                                    label2, len(aug_indexes[-1])))
                    aug_indexes = np.concatenate(aug_indexes)
                    tr = np.append(tr, aug_indexes)
                    info("Total size change: from {} to {} training instances".
                         format(orig_tr_size, len(tr)))
                elif self.sampling_method == defs.sampling.undersample:
                    for (label1, label2, ratio) in self.sampling_ratios:
                        aug_indexes.append(
                            ldaug.undersample_to_ratio(self.train_data,
                                                       self.train_labels,
                                                       [label1, label2],
                                                       ratio,
                                                       only_indexes=True))
                        info(
                            "Sampled via {}, to ratio {}, for labels {},{}. Modification size: {} instances."
                            .format(self.sampling_method, ratio, label1,
                                    label2, len(aug_indexes[-1])))
                    aug_indexes = np.concatenate(aug_indexes)
                    tr = np.delete(tr, aug_indexes)
                    info("Total size change: from {} to {} training instances".
                         format(orig_tr_size, len(tr)))
                else:
                    error(
                        "Undefined augmentation method: {} -- available are {}"
                        .format(self.sampling_method, defs.avail_sampling))
                splits[split_index] = (tr, vl)

        # save and return the splitter splits
        makedirs(dirname(trainval_serialization_file), exist_ok=True)
        write_pickled(trainval_serialization_file, splits)
        return splits

    # # apply required preprocessing on data and labels for a run
    # def preprocess_data_labels(self, train, val, test):
    #     train_data, train_labels = train
    #     train = self.process_input(train_data), train_labels

    #     if val is not None:
    #         val_data, val_labels = val
    #         val = self.process_input(val_data), val_labels

    #     test_data, test_labels = test
    #     test = self.process_input(test_data), test_labels
    #     return train, val, test

    # handle multi-vector items, expanding indexes to the specified sequence length
    # def expand_index_to_sequence(self, fold_data):
    #     # map to indexes in the full-sequence data (e.g. times sequence_length)
    #     # fold_data = list(map(lambda x: x * self.sequence_length if len(x) > 0 else np.empty((0,)), fold_data))
    #     for i in range(len(fold_data)):
    #         if fold_data[i] is None:
    #             continue
    #         # expand with respective sequence members (add an increment, vstack)
    #         # reshape to a single vector, in the vertical (column) direction, that increases incrementally
    #         fold_data[i] = tuple([
    #             np.ndarray.flatten(np.vstack([
    #                 (x * self.sequence_length) +
    #                 incr if x is not None else None
    #                 for incr in range(self.sequence_length)
    #             ]),
    #                                order='F') for x in fold_data[i]
    #         ])
    #     return fold_data

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
        self.do_traintest()

        self.outputs.set_vectors(Vectors(vecs=self.evaluator.predictions))

    def process_component_inputs(self):
        # get data and labels
        error("Learner needs at least two-input bundle input list.",
              type(self.inputs) is not BundleList)
        error("{} needs vector information.".format(self.component_name),
              not self.inputs.has_vectors())
        error("{} needs label information.".format(self.component_name),
              not self.inputs.has_labels())

        self.train_index, self.test_index = (np.squeeze(np.asarray(x)) for x in self.inputs.get_indices(single=True).instances)
        self.embeddings = self.inputs.get_vectors(single=True).instances
        self.train_labels, self.test_labels = self.inputs.get_labels(single=True).instances

    def load_existing_predictions(self, current_test_instance_indexes, current_test_labels):
        # get predictions and instance indexes they correspond to
        existing_predictions, existing_instance_indexes = self.get_existing_predictions()
        if existing_predictions is not None:
            info("Loaded existing predictions.")
            error(
                "Different instance indexes loaded than the ones generated.",
                not np.all(
                    np.equal(existing_instance_indexes, current_test_instance_indexes)))
            existing_test_labels = self.validation.get_test_labels(
                test_instance_indexes)
            error(
                "Different instance labels loaded than the ones generated.",
                not np.all(
                    np.equal(existing_test_labels, current_test_labels)))
        return existing_predictions, existing_instance_indexes


    def get_data_from_index(self, index, embeddings):
        """Get data index from the embedding matrix"""
        if np.squeeze(index).ndim > 1:
            # if we have multi-element index, there has to be an aggregation method defined for the learner.
            error("The learner [{}] has no defined aggregation and is not sequence-capable, but the input index has shape {}".
            format(self.name, index.shape), self.input_aggregation is None and self.sequence_length < 2)
        return embeddings[index] if len(index) > 0 else None

    class ValidatonSetting:
        def __init__(self, folds, portion, test_present, do_multilabel):
            self.label_indexes = None
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
                error(
                    "Mismatch between expected folds ({}) and loaded data of {} splits."
                    .format(self.folds, len(idx)),
                    len(idx) != self.folds)
            elif self.do_portion:
                info("Loaded train/val split of {} / {}.".format(
                    *list(map(len, idx[0]))))

        def get_model_path(self, base_path):
            return self.modify_suffix(base_path) + ".model"

        def conclude_iteration(self):
            if self.do_folds:
                self.current_fold += 1

        def modify_suffix(self, base_path):
            if self.do_folds:
                return base_path + "_fold{}".format(self.current_fold)
            elif self.do_portion:
                base_path += "_valportion{}".format(self.portion)
            return base_path

        def assign_data(self, embeddings, train_index, train_labels, test_labels, test_index):
            self.embeddings = embeddings
            self.train_index = train_index
            self.test_index = test_index
            self.train_labels, self.test_labels = train_labels, test_labels

        # get training, validation, test data chunks, given the input indexes and validation setting
        def get_run_data(self, iteration_index, trainval_idx):
            """get training and validation data chunks, given the input indexes"""
            if self.do_folds:
                error(
                    "Iteration index: {} / fold index: {} mismatch in validation coordinator."
                    .format(iteration_index, self.current_fold),
                    iteration_index != self.current_fold)
            train_idx, val_idx = trainval_idx

            train_labels, val_labels = self.get_trainval_labels(iteration_index, trainval_idx)

            if len(train_idx) > 0:
                if not self.do_multilabel:
                    train_labels = np.squeeze(np.asarray(train_labels))
                curr_train_idx = np.squeeze(np.asarray([self.train_index[idx] for idx in train_idx]))

            if len(val_idx) > 0:
                if not self.do_multilabel:
                    val_labels = np.squeeze(np.asarray(val_labels))
                curr_val_idx = np.squeeze(np.asarray([self.train_index[idx] for idx in val_idx]))
            else:
                curr_val_idx = None

            if self.use_validation_for_testing:
                curr_test_idx = curr_val_idx
                test_labels = val_labels
                curr_val_idx, val_labels = None, None
            else:
                curr_test_idx, test_labels = self.test_index, self.test_labels

            if len(val_idx) > 0 and self.use_validation_for_testing:
                # mark the test instance indexes as the val. indexes of the train
                instance_indexes = val_idx
            else:
                instance_indexes = range(len(self.test_index))
            return curr_train_idx, train_labels, curr_val_idx, val_labels, curr_test_idx, test_labels, instance_indexes

        def get_test_labels(self, instance_indexes):
            if self.use_validation_for_testing:
                return self.train_labels[instance_indexes]
            else:
                error(
                    "Non-full instance indexes encountered, but validation is not set to act as testing",
                    instance_indexes != range(len(self.test_labels)))
                return self.test_labels

        def set_trainval_label_index(self, trainval_idx):
            self.label_indexes = trainval_idx

        def get_trainval_labels(self, iteration_index, trainval_idx):
            if self.label_indexes is not None:
                train_idx, val_idx = self.label_indexes[iteration_index]
            else:
                train_idx, val_idx = trainval_idx
            return [[np.asarray(self.train_labels[i])
                     for i in idx] if idx is not None else None
                    for idx in [train_idx, val_idx]]

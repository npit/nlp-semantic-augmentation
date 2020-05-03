import numpy as np

from utils import error, info


class ValidationSetting:
    def __init__(self, folds, portion, test_present, use_labels=False, do_multilabel=False):
        """Constructor for the validation setting class

        :param folds: int, The number of folds to train with
        :param portion: float, The percent portion
        :param test_present: boolean, Whether test data are available
        :param use_labels: boolean, Whether labels are available
        :param do_multilabel: boolean, Whether to do a multli-label run
        :returns: 
        :rtype: 

        """
        self.test_instance_indexes = None
        self.label_indexes = None
        self.do_multilabel = do_multilabel
        self.use_labels = use_labels
        self.do_folds = folds is not None
        self.folds = folds
        self.portion = portion
        self.do_portion = portion is not None
        self.use_validation = self.do_folds or self.do_portion
        self.use_for_testing = self.use_validation and not test_present
        if self.do_folds:
            self.descr = "{} stratified folds".format(self.folds)
            self.current_fold = 0
        elif self.do_portion:
            self.descr = "{} validation portion".format(self.portion)
        else:
            self.descr = "(no validation)"
        self.empty = np.empty((0,), np.int32)

    def __str__(self):
        """String override
        """
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
            if len(idx) != self.folds:
                error("Mismatch between expected folds ({self.folds}) and loaded data of {len(idx)} splits.")
        elif self.do_portion:
            info("Loaded train/val split of {} / {}.".format(*list(map(len, idx[0]))))

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

    def assign_data(self, embeddings, train_index, train_labels=None, test_labels=None, test_index=None):
        self.embeddings = embeddings
        self.train_embedding_index = train_index
        self.test_embedding_index = test_index
        self.train_labels, self.test_labels = train_labels, test_labels
        if test_index is not None:
            self.test_instance_indexes = np.arange(len(test_index))

    def get_run_labels(self, iteration_index, trainval_idx):
        """Fetch current labels, if defined."""
        if not self.use_labels:
            return None, None, None
        train_labels, val_labels = self.get_trainval_labels(iteration_index, trainval_idx)

        # use validation labels for testing
        if self.use_for_testing:
            test_labels, val_labels = val_labels, []
        else:
            test_labels = self.test_labels

        # if single-label, squeeze to ndarrays
        if not self.do_multilabel:
            train_labels = np.squeeze(np.asarray(train_labels))
            val_labels = np.squeeze(np.asarray(val_labels))
            test_labels = np.squeeze(np.asarray(test_labels))

        return train_labels, val_labels, test_labels

    def get_test_data(self):
        """Retrieve indexes to master test embedding indexes, and instance indexes
        """
        return (self.test_embedding_index, self.test_instance_indexes)

    # get training, validation, test data chunks, given the input indexes and the validation setting
    def get_run_data(self, iteration_index, trainval_idx):
        """Function to retrieve training and validation instance indexes.

        Given current input validation indexes, retrieve the indexes to embeddings.

        :param iteration_index: int, Training (fold) iteration.
        :param trainval_idx: tuple, Training & validation index
        """
        if not self.use_validation:
            return (self.train_embedding_index, self.empty, *self.get_test_data())

        if self.do_folds:
            if iteration_index != self.current_fold:
                error("Iteration index: {iteration_index} / fold index: {self.current_fold} mismatch in validation coordinator.")

        train_idx, val_idx = trainval_idx

        curr_train_idx = self.train_embedding_index[train_idx]
        curr_val_idx = self.train_embedding_index[val_idx]

        if self.use_for_testing:
            # zero out validation, swap with test
            curr_test_idx, instance_indexes = curr_val_idx, val_idx
            curr_val_idx = self.empty
        else:
            # get regular test indexes
            curr_test_idx, instance_indexes = self.get_test_data()
        return curr_train_idx, curr_val_idx, curr_test_idx, instance_indexes

    def get_test_labels(self, instance_indexes):
        """Retrieve the test labels based in the input instance indexes

        :param instance_indexes: 
        :returns: 
        :rtype: 

        """
        if self.use_validation and self.use_for_testing:
            # use the instance indexes for the training labelset
            return self.train_labels[instance_indexes]

        # in all other scenarios, the entirety of the test labels is required
        # make sure entirety of instance indexes is requested
        error("Non-full instance indexes encountered, but no validation setting is defined!", instance_indexes != self.test_instance_indexes)
        # retrieve the entirety of the test labels
        return self.test_labels

    def set_trainval_label_index(self, trainval_idx):
        self.label_indexes = trainval_idx

    def get_trainval_labels(self, iteration_index, trainval_idx):
        """Retrieve the training/validation labels, give the current validation iteration and input indexes

        :param iteration_index: 
        :param trainval_idx: 
        :returns: 
        :rtype: 

        """
        if not self.use_validation:
            return self.train_labels, self.empty

        print("TODO CHECK")
        if self.label_indexes is not None:
            train_idx, val_idx = self.label_indexes[iteration_index]
        else:
            train_idx, val_idx = trainval_idx

        train_labels = [np.asarray(self.train_labels[i]) for i in train_idx]
        val_labels = [np.asarray(self.train_labels[i]) for i in val_idx]
        return train_labels, val_labels

import numpy as np

from utils import error, info


class ValidationSetting:
    def __init__(self, folds, portion, test_present, use_labels=False, do_multilabel=False):
        self.label_indexes = None
        self.do_multilabel = do_multilabel
        self.use_labels = use_labels
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

    def assign_data(self, embeddings, train_index, train_labels=None, test_labels=None, test_index=None):
        self.embeddings = embeddings
        self.train_index = train_index
        self.test_index = test_index
        self.train_labels, self.test_labels = train_labels, test_labels

    def get_run_labels(self, iteration_index, trainval_idx):
        """Fetch current labels, if defined."""
        if not self.use_labels:
            return None, None, None
        train_labels, val_labels = self.get_trainval_labels(iteration_index, trainval_idx)

        # use validation labels for testing
        if self.use_validation_for_testing:
            val_labels, test_labels = val_labels, None
        else:
            test_labels = self.test_labels

        # if single-label, squeeze to ndarrays
        if not self.do_multilabel:
            train_labels = np.squeeze(np.asarray(train_labels))
            test_labels = np.squeeze(np.asarray(test_labels))
            if len(val_idx) > 0 and not self.use_validation_for_testing:
                val_labels = np.squeeze(np.asarray(val_labels))

        return train_labels, val_labels, test_labels

    # get training, validation, test data chunks, given the input indexes and validation setting
    def get_run_data(self, iteration_index, trainval_idx):
        """get training and validation data chunks, given the input indexes"""
        if self.do_folds:
            error(
                "Iteration index: {} / fold index: {} mismatch in validation coordinator."
                .format(iteration_index, self.current_fold),
                iteration_index != self.current_fold)
        train_idx, val_idx = trainval_idx

        ### train_labels, val_labels = self.get_trainval_labels(iteration_index, trainval_idx)

        # if len(train_idx) > 0:
        #     if not self.do_multilabel:
        #         train_labels = np.squeeze(np.asarray(train_labels))
        #     curr_train_idx = np.squeeze(np.asarray([self.train_index[idx] for idx in train_idx]))
        curr_train_idx = np.squeeze(np.asarray([self.train_index[idx] for idx in train_idx]))

        if len(val_idx) > 0:
            # if not self.do_multilabel:
            #     val_labels = np.squeeze(np.asarray(val_labels))
            curr_val_idx = np.squeeze(np.asarray([self.train_index[idx] for idx in val_idx]))
        else:
            curr_val_idx = None

        if self.use_validation_for_testing:
            curr_test_idx = curr_val_idx
            instance_indexes = val_idx
            # test_labels = val_labels
            # curr_val_idx, val_labels = None, None
        else:
            # curr_test_idx, test_labels = self.test_index, self.test_labels
            curr_test_idx= self.test_index
            instance_indexes = range(len(self.test_index))
        # if not self.do_multilabel:
        #     test_labels = np.squeeze(np.asarray(test_labels))

        # if len(val_idx) > 0 and self.use_validation_for_testing:
        #     # mark the test instance indexes as the val. indexes of the train
        #     instance_indexes = val_idx
        # else:
        #     instance_indexes = range(len(self.test_index))

        return curr_train_idx, curr_val_idx, curr_test_idx, instance_indexes

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

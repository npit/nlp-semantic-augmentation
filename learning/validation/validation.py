import numpy as np
from learning.validation.splitting import kfold_split, portion_split

from utils import error, info, warning, write_pickled, read_pickled

class ValidationSetting:
    train_idx, test_idx, test_idx = None, None, None
    labels, label_info = None, None
    seed = None
    folds, portion = None, None
    current_fold = None

    test_via_validation = False


    def __init__(self, config, train_idx, test_idx, labels=None, label_info=None, folds=None, portion=None, seed=1337):
        """Constructor"""
        self.config = config
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.labels = labels
        self.label_info = label_info
        self.folds = folds
        self.portion = portion
        self.seed = seed
        self.make_splits()

    def make_splits(self):
        """Produce validation splits, if defined"""
        # produce fold/portion splits of the training indexes: these output indexes to the tr. indexes themselves
        if self.folds is not None:
            meta_trainval_idx = kfold_split(self.train_idx, self.folds, self.seed, self.labels, self.label_info)
        elif self.portion is not None:
            meta_trainval_idx = portion_split(self.train_idx, self.portion, self.seed, self.labels, self.label_info)
        else:
            meta_trainval_idx = [(np.arange(len(self.train_idx)), np.arange(0, dtype=np.int32))]
        # "dereference" the metaindexes to point to the data themselves
        self.trainval_idx = []
        for (tidx, vidx) in meta_trainval_idx:
            self.trainval_idx.append((self.train_idx[tidx], self.train_idx[vidx]))

    def get_total_iterations(self):
        return len(self.trainval_idx)

    def reserve_validation_for_testing(self):
        """Use the validation indexes to test the model"""
        info("Reserving validation data for testing.")
        if self.test_idx is not None and self.test_idx.size > 0:
            warning(f"Reserving validation for testing but {len(self.test_idx)} test index exists!")
            warning(f"Deleting existing test indexes")
        self.test_idx = []
        self.test_via_validation = True
        for i in range(len(self.trainval_idx)):
            # fetch and replace the validation index
            val_idx = self.trainval_idx[i][1]
            self.trainval_idx[i] = (self.trainval_idx[i][0], np.arange(0, dtype=np.int32))
            # add it as the test index
            self.test_idx.append(val_idx)

    def get_trainval_indexes(self):
        return self.trainval_idx

    def get_train_indexes(self):
        return [x[0] for x in self.trainval_idx]

    def get_val_indexes(self):
        return [x[1] for x in self.trainval_idx]

    def get_test_indexes(self):
        ti = self.test_idx
        if type(self.test_idx) is not list:
            ti = [self.test_idx]
        if self.folds is not None:
            ti *= self.folds
        return ti

    def get_info_string(self):
        return get_info_string(self.config)

    def write_trainval(self, output_file):
        write_pickled(output_file, (self.trainval_idx, self.test_idx), "validation trainval indexes") 


def get_info_string(config):
    """Fetch validation-related information in a string"""
    if config.train.folds is not None:
        return f"folds_{config.train.folds}"
    elif config.train.validation_portion is not None:
        return "valportion_{}".format(config.train.validation_portion)
    return ""

def load_trainval(path):
    dat = read_pickled(path)
    trainval_idx = dat[0]
    train = trainval_idx[0]
    val = trainval_idx[1]
    test = dat[1]
    return train, val, test

# class ValidationSetting:
#     """Class to configure validation
#     Possible scenarios:

#     No validation: train on training data, test on test data, if they exist

#     K-fold cross validation, no test data supplied: Classic cross-validation on the training data.
#     K-fold cross validation, test data supplied: Split training data to K folds, train K models, test each on the test data.

#     K % validation portion, no test data supplied: Train on (100-k)%, test on remaining k%
#     K % validation portion, test data supplied: Train on (100-k)%, test on the test set

#     If numeric labels are supplied, splitting is stratified.
#     """



#     def __init__(self, folds, portion, test_present, use_labels=False, do_multilabel=False):
#         """Constructor for the validation setting class

#         :param folds: int, The number of folds to train with
#         :param portion: float, The percent portion
#         :param test_present: boolean, Whether test data are available
#         :param use_labels: boolean, Whether labels are available
#         :param do_multilabel: boolean, Whether to do a multli-label run
#         :returns: 
#         :rtype: 

#         """
#         self.test_instance_indexes = None
#         self.label_indexes = None
#         self.do_multilabel = do_multilabel
#         self.use_labels = use_labels
#         self.do_folds = folds is not None
#         self.folds = folds
#         self.portion = portion
#         self.do_portion = portion is not None
#         self.use_validation = self.do_folds or self.do_portion
#         self.use_for_testing = self.use_validation and not test_present
#         if self.do_folds:
#             self.descr = "{} stratified folds".format(self.folds)
#             self.current_fold = 0
#         elif self.do_portion:
#             self.descr = "{} validation portion".format(self.portion)
#         else:
#             self.descr = "(no validation)"
#         self.empty = np.empty((0,), np.int32)

#     def __str__(self):
#         """String override
#         """
#         return self.descr

#     def get_current_descr(self):
#         if self.do_folds:
#             return "fold {}/{}".format(self.current_fold + 1, self.folds)
#         elif self.do_portion:
#             return "{}-val split".format(self.portion)
#         else:
#             return "(no-validation)"

#     def check_indexes(self, idx):
#         if self.do_folds:
#             if len(idx) != self.folds:
#                 error("Mismatch between expected folds ({self.folds}) and loaded data of {len(idx)} splits.")
#         elif self.do_portion:
#             info("Loaded train/val split of {} / {}.".format(*list(map(len, idx[0]))))

#     def get_model_path(self, base_path):
#         return self.modify_suffix(base_path) + ".model"

#     def conclude_iteration(self):
#         if self.do_folds:
#             self.current_fold += 1

#     def get_info_string(self):
#         if self.do_folds:
#             return "_fold{}".format(self.current_fold)
#         elif self.do_portion:
#             return "_valportion{}".format(self.portion)
#         return ""

#     def assign_data(self, embeddings, train_index, train_labels=None, test_labels=None, test_index=None):
#         self.embeddings = embeddings
#         self.train_embedding_index = train_index
#         self.test_embedding_index = test_index
#         self.train_labels, self.test_labels = train_labels, test_labels
#         if test_index is not None:
#             self.test_instance_indexes = np.arange(len(test_index))

#     def get_run_labels(self, iteration_index, trainval_idx):
#         """Fetch current labels, if defined."""
#         if not self.use_labels:
#             return None, None, None
#         train_labels, val_labels = self.get_trainval_labels(iteration_index, trainval_idx)

#         # use validation labels for testing
#         if self.use_for_testing:
#             test_labels, val_labels = val_labels, []
#         else:
#             test_labels = self.test_labels

#         # if single-label, squeeze to ndarrays
#         if not self.do_multilabel:
#             train_labels = np.squeeze(np.asarray(train_labels))
#             val_labels = np.squeeze(np.asarray(val_labels))
#             test_labels = np.squeeze(np.asarray(test_labels))

#         return train_labels, val_labels, test_labels

#     def get_test_data(self):
#         """Retrieve indexes to master test embedding indexes, and instance indexes
#         """
#         return (self.test_embedding_index, self.test_instance_indexes)

#     # get training, validation, test data chunks, given the input indexes and the validation setting
#     def get_run_data(self, iteration_index, trainval_idx):
#         """Function to retrieve training and validation instance indexes.

#         Given current input validation indexes, retrieve the indexes to embeddings.

#         :param iteration_index: int, Training (fold) iteration.
#         :param trainval_idx: tuple, Training & validation index
#         """
#         if not self.use_validation:
#             return (self.train_embedding_index, self.empty, *self.get_test_data())

#         if self.do_folds:
#             if iteration_index != self.current_fold:
#                 error("Iteration index: {iteration_index} / fold index: {self.current_fold} mismatch in validation coordinator.")

#         train_idx, val_idx = trainval_idx

#         curr_train_idx = self.train_embedding_index[train_idx]
#         curr_val_idx = self.train_embedding_index[val_idx]

#         if self.use_for_testing:
#             # zero out validation, swap with test
#             curr_test_idx, instance_indexes = curr_val_idx, val_idx
#             curr_val_idx = self.empty
#         else:
#             # get regular test indexes
#             curr_test_idx, instance_indexes = self.get_test_data()
#         self.curr_test_idx = curr_test_idx
#         return curr_train_idx, curr_val_idx, curr_test_idx, instance_indexes

#     def get_test_labels(self, instance_indexes):
#         """Retrieve the test labels based in the input instance indexes

#         :param instance_indexes: 
#         :returns: 
#         :rtype: 

#         """
#         if self.use_validation and self.use_for_testing:
#             # use the instance indexes for the training labelset
#             return self.train_labels[instance_indexes]

#         # in all other scenarios, the entirety of the test labels is required
#         # make sure entirety of instance indexes is requested
#         error("Non-full instance indexes encountered, but no validation setting is defined!", instance_indexes != self.test_instance_indexes)
#         # retrieve the entirety of the test labels
#         return self.test_labels

#     def set_trainval_label_index(self, trainval_idx):
#         self.label_indexes = trainval_idx

#     def get_trainval_labels(self, iteration_index, trainval_idx):
#         """Retrieve the training/validation labels, give the current validation iteration and input indexes

#         :param iteration_index: 
#         :param trainval_idx: 
#         :returns: 
#         :rtype: 

#         """
#         if not self.use_validation:
#             return self.train_labels, self.empty

#         print("TODO CHECK")
#         if self.label_indexes is not None:
#             train_idx, val_idx = self.label_indexes[iteration_index]
#         else:
#             train_idx, val_idx = trainval_idx

#         train_labels = [np.asarray(self.train_labels[i]) for i in train_idx]
#         val_labels = [np.asarray(self.train_labels[i]) for i in val_idx]
#         return train_labels, val_labels

#     def get_information(self):
#         """Retrieve the validation setting"""
#         pass

#     def get_prediction_indexes(self):
#         """Get the indexes that correspond to prediction generation"""
#         return self.curr_test_idx
#         # if self.use_for_testing:
#         #     # return val indexes
#         #     return self.curr_val_idx
#         # else:
#         #     return self.curr_test_idx

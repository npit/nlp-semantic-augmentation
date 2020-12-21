from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split
import numpy as np

from learning.sampling import oversample_single_sample_labels

from utils import info, one_hot, warning, error
"""Module for validation splits"""



def kfold_split(data, num_folds, seed, labels=None, label_info=None):
    """Do K-fold cross-validation"""
    num_data = len(data)
    info(f"Splitting {num_data} input data to {num_folds} folds")
    if labels is None:
        return list(KFold(num_folds, shuffle=True, random_state=seed).split(data))
    else:
        multilabel = label_info.multilabel
        labelset = label_info.labelset
        if multilabel:
            splitter = IterativeStratification(num_folds, order=1)
            oh_labels = one_hot(labels, len(labelset), is_multilabel=True)
            return list(splitter.split(np.zeros(len(labels)), oh_labels))
        else:
            # if False:
            #     import ipdb; ipdb.set_trace()
            #     data, labels = oversample_single_sample_labels(data, np.concatenate(labels), target_num=num_folds)
            try:
                return list(StratifiedKFold(num_folds, shuffle=True, random_state=seed).split(data, labels))
            except ValueError as ve:
                error(f"Unable to complete a stratified fold split: {ve}")
                # return kfold_split(data, num_folds, seed, labels=None, label_info=None)



def portion_split(data, portion, seed=1337, labels=None, label_info=None):
    """Perform a k% split to train-validation instances"""

    if labels is None:
            info(f"Portion-splitting with input data: {len(data)} samples on a {portion} validation portion")
            return list(ShuffleSplit( n_splits=1, test_size=portion, random_state=seed).split(data))
    else:
        multilabel = label_info.multilabel
        labelset = label_info.labelset
        if multilabel:
            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[portion, 1.0-portion])
            labels = one_hot(labels, len(labelset), True)
            train_indexes, test_indexes = next(stratifier.split(np.zeros(len(data)), labels))
            return [(train_indexes, test_indexes)]
        else:
            # if False:
            #     import ipdb; ipdb.set_trace()
            #     data, labels = oversample_single_sample_labels(data, np.concatenate(labels), target_num=2)
            #     test_size = np.floor(portion * len(data))
            #     lset = len(label_info.labelset)
            #     if test_size < lset:
            #         new_portion = np.ceil(lset / len(data) * 1000) / 1000
            #         warning(f"Setting portion from {portion} (which results in {test_size} test data) to {new_portion} since we have {lset} labels")
            #         portion = new_portion
            try:
                return list(StratifiedShuffleSplit(n_splits=1, test_size=portion, random_state=seed).split(data, labels))
            except ValueError as ve:
                error(f"Unable to complete a stratified split: {ve}")
                # import ipdb; ipdb.set_trace()
                # return portion_split(data, portion, seed, labels=None, label_info=None)

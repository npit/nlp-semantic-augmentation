from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split
import numpy as np

# from learning.sampling import oversample_single_sample_labels

from utils import info, one_hot, warning, error
"""Module for validation splits"""



def kfold_split(data, num_folds, seed, labels=None, label_info=None):
    """Do K-fold cross-validation"""
    num_data = len(data)
    msg = f"Splitting {num_data} input data to {num_folds} folds"
    if labels is None:
        info(msg)
        return list(KFold(num_folds, shuffle=True, random_state=seed).split(data))
    else:
        multilabel = label_info.multilabel
        labelset = label_info.labelset
        if multilabel:
            info(msg +" using iterative stratification.")
            splitter = IterativeStratification(num_folds, order=1)
            oh_labels = one_hot(labels, len(labelset), is_multilabel=True)
            return list(splitter.split(np.zeros(len(labels)), oh_labels))
        else:
            try:
                info(msg +" using stratification.")
                return list(StratifiedKFold(num_folds, shuffle=True, random_state=seed).split(data, labels))
            except ValueError as ve:
                error(f"Unable to complete a stratified fold split: {ve}")
                # return kfold_split(data, num_folds, seed, labels=None, label_info=None)



def portion_split(data, portion, seed=1337, labels=None, label_info=None):
    """Perform a k% split to train-validation instances"""

    msg = f"Portion-splitting with input data: {len(data)} samples on a {portion} validation portion"
    if labels is None:
            info(msg)
            return list(ShuffleSplit( n_splits=1, test_size=portion, random_state=seed).split(data))
    else:
        multilabel = label_info.multilabel
        labelset = label_info.labelset
        if multilabel:
            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[portion, 1.0-portion])
            labels = one_hot(labels, len(labelset), True)
            info(msg +" using iterative stratification.")
            train_indexes, test_indexes = next(stratifier.split(np.zeros(len(data)), labels))
            return [(train_indexes, test_indexes)]
        else:
            try:
                info(msg +" using stratification.")
                return list(StratifiedShuffleSplit(n_splits=1, test_size=portion, random_state=seed).split(data, labels))
            except ValueError as ve:
                error(f"Unable to complete a stratified split: {ve}")
                # import ipdb; ipdb.set_trace()
                # return portion_split(data, portion, seed, labels=None, label_info=None)

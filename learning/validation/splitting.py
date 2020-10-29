from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split
import numpy as np

from utils import info, one_hot
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
            return list(StratifiedKFold(num_folds, shuffle=True, random_state=seed).split(data, labels))



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
            breakpoint()
            return list(StratifiedShuffleSplit(n_splits=1, test_size=portion, random_state=seed).split(data, labels))

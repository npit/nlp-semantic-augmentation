"""Class implementing dataset subsampling"""
import random

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils import align_index, debug, error, info, warning


class Sampler:
    def __init__(self, config):
        self.dlim = config.dataset.data_limit
        self.clim = config.dataset.class_limit

    def get_limited_name(self, name):
        """Modify name to reflect data / class limiting"""
        if self.dlim:
            dlim = self.dlim
            dlim = dlim + [-1] if len(dlim) == 1 else dlim
            dlim = [x if x >= 0 else None for x in dlim]
            # make limited name like datasetname_dlim_tr100_te50
            # for train, test limiting of 100 and 50 instances, resp.
            name += "_dlim"
            for l, n in zip(dlim, "tr te".split()):
                name += "_{}{}".format(n, l)
        if self.clim:
            clim = self.clim
            name += "_clim_{}".format(clim)
        return name


    # region # limiters
    def limit_data_stratify(self, num_limit, data, labels):
        """Label distribution-aware data limiting"""
        limit_ratio = num_limit / len(data)
        splitter = StratifiedShuffleSplit(1, test_size=limit_ratio)
        splits = list(splitter.split(np.zeros(len(data)), labels))
        data = [data[n] for n in splits[0][1]]
        labels = [labels[n] for n in splits[0][1]]

        # fix up any remaining inconsistency
        while not len({num_limit, len(data), len(labels)}) == 1:
            # get label, label_indexes tuple list
            counts = [(x, [i for i in range(len(labels)) if x == labels[i]]) for x in labels]
            # get most populous label
            maxfreq_label, maxfreq_label_idx = max(counts, key=lambda x: len(x[1]))
            # drop one from it
            idx = random.choice(maxfreq_label_idx)
            del data[idx]
            del labels[idx]
            # remove by index of index
            idx_idx = maxfreq_label_idx.index(idx)
            del maxfreq_label_idx[idx_idx]
        return data, np.asarray(labels)

    def limit_data_simple(self, num_limit, data, labels=None):
        """Simple data limiter"""
        idxs = random.sample(list(range(len(data))), num_limit)
        data = [data[i] for i in idxs]
        if labels is not None:
            labels = np.asarray([labels[i] for i in idxs])
        return data, labels

    def apply_data_limit(self, data, labels=None):
        lim = self.dlim
        lim = lim + [-1] if len(lim) == 1 else lim
        lim = [x if x >= 0 else None for x in lim]
        ltrain, ltest = lim
        train, test = data
        if labels:
            train_labels, test_labels = labels
        else:
            train_labels, test_labels = None, None
        if ltrain:
            if len(train) < ltrain:
                error("Attempted to data-limit {} train items to {}".format(len(train), ltrain))
            if labels:
                # use stratification
                train, train_labels = self.limit_data_stratify(ltrain, train, train_labels)
                info("Limited loaded data to {} train items.".format(len(train)))
                error("Inconsistent limiting in train data.", len({ltrain, len(train), len(train_labels)}) != 1)
            else:
                warning("Resorting to non-stratified trainset limiting")
                train, train_labels = self.limit_data_simple(ltrain, train, train_labels)

        if ltest and test:
            error("Attempted to data-limit {} test items to {}".format(len(test), ltest), len(test) < ltest)
            if labels:
                # use stratification
                test, test_labels = self.limit_data_stratify(ltest, test, test_labels)
                info("Limited loaded data to {} test items.".format(len(test)))
                error("Inconsistent limiting in test data.", len({ltest, len(test), len(test_labels)}) != 1)
            else:
                warning("Resorting to non-stratified testset limiting")
                test, test_labels = self.limit_data_simple(ltest, test, test_labels)
        if labels:
            return (train, test), (train_labels, test_labels)
        return (train, test), labels

    def restrict_to_classes(self, data, labels, restrict_classes):
        new_data, new_labels = [], []
        # ensure iterables
        restore_ndarray = False
        if type(labels) is np.ndarray:
            labels = [[l] for l in labels]
            restore_ndarray = True
        for d, l in zip(data, labels):
            valid_classes = [cl for cl in l if cl in restrict_classes]
            if not valid_classes:
                continue
            new_data.append(d)
            new_labels.append(valid_classes)

        if restore_ndarray:
            new_labels = np.asarray([x for l in new_labels for x in l])
        return new_data, new_labels

    def apply_class_limit(self, data, labels, labelset, label_names, multilabel):
        c_lim = self.clim
        num_labels = len(labelset)
        error("Specified non-sensical class limit from {} classes to {}.".format(num_labels, c_lim), c_lim >= num_labels)
        if data:
            train, test = data
            train_labels, test_labels = labels
            # data have been loaded -- apply limit
            retained_classes = random.sample(labelset, c_lim)
            info("Limiting to the {}/{} classes: {}".format(c_lim, num_labels, retained_classes))
            if multilabel:
                debug("Max train/test labels per item prior: {} {}".format(max(map(len, train_labels)), max(map(len, test_labels))))
            train, train_labels = self.restrict_to_classes(train, train_labels, retained_classes)
            test, test_labels = self.restrict_to_classes(test, test_labels, retained_classes)
            num_labels = len(retained_classes)
            if not num_labels:
                error("Zero labels after limiting.")
            # remap retained classes to indexes starting from 0
            train_labels = align_index(train_labels, retained_classes)
            test_labels = align_index(test_labels, retained_classes)
            # fix the label names
            label_names = [label_names[rc] for rc in retained_classes]
            labelset = retained_classes
            if multilabel:
                debug("Max train/test labels per item post: {} {}".format(max(map(len, train_labels)), max(map(len, test_labels))))
        return (train, test), (train_labels, test_labels), (labelset, label_names)

    def subsample(self, data, labels=None, labelset=None, label_names=None, multilabel=None):
        """Apply data and class sub-sampling"""
        if any(l is not None for l in labels) and self.clim is not None:
            data, labels, (labelset, label_names) = self.apply_class_limit(data, labels, labelset, label_names, multilabel)
        data, labels = self.apply_data_limit(data, labels)
        return data, labels, labelset, label_names

    # endregion

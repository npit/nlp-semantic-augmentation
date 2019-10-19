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

    def limit_data_stratify(num_limit, data, labels):
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




    # def limit_data_simple(num_limit, data):
    #     idxs = random.sample(list(range(len(data))), num_limit)
    #     data = [data[i] for i in idxs]
    #     return data

    def limit_data_simple(self, num_limit, data, labels=None):
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
                error("Inconsistent limiting in train data.",  len({ltrain, len(train), len(train_labels)}) != 1)
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
        return (train, test)

    def restrict_to_classes(self, data, labels, restrict_classes):
        new_data, new_labels = [], []
        for d, l in zip(data, labels):
            valid_classes = [cl for cl in l if cl in restrict_classes]
            if not valid_classes:
                continue
            new_data.append(d)
            new_labels.append(valid_classes)
        return new_data, new_labels

    def apply_class_limit(self, data, labels):
        c_lim = self.clim
        num_labels = len(set(labels))
        error("Specified non-sensical class limit from {} classes to {}.".format(num_labels, c_lim), c_lim >= num_labels)
        if c_lim is not None:
            if self.train:
                # data have been loaded -- apply limit
                retained_classes = random.sample(list(range(num_labels)), c_lim)
                info("Limiting to the {}/{} classes: {}".format(c_lim, num_labels, retained_classes))
                if self.multilabel:
                    debug("Max train/test labels per item prior: {} {}".format(max(map(len, self.train_labels)), max(map(len, self.test_labels))))
                self.train, self.train_labels = self.restrict_to_classes(self.train, self.train_labels, retained_classes)
                self.test, self.test_labels = self.restrict_to_classes(self.test, self.test_labels, retained_classes)
                num_labels = len(retained_classes)
                if not num_labels:
                    error("Zero labels after limiting.")
                # remap retained classes to indexes starting from 0
                self.train_labels = align_index(self.train_labels, retained_classes)
                self.test_labels = align_index(self.test_labels, retained_classes)
                # fix the label names
                self.train_label_names = [self.train_label_names[rc] for rc in retained_classes]
                self.test_label_names = [self.test_label_names[rc] for rc in retained_classes]
                if self.multilabel:
                    debug("Max train/test labels per item post: {} {}".format(max(map(len, self.train_labels)), max(map(len, self.test_labels))))
        return (train, test), (train_labels, test_labels)

    def subsample(self, data, labels=None):
        """Apply data and class sub-sampling"""
        if labels is not None:
            data, labels = self.apply_class_limit(data, labels)
        return self.apply_data_limit(data, labels)

    # endregion

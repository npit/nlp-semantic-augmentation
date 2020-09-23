"""Class implementing dataset subsampling"""
import random

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils import align_index, debug, error, info, warning, get_labelset


class Sampler:
    def __init__(self, config):
        self.dlim = config.data_limit
        self.clim = config.class_limit

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
    def limit_data_stratify(self, num_limit, data, labels, targets):
        """Label distribution-aware data limiting"""
        limit_ratio = num_limit / len(data)
        splitter = StratifiedShuffleSplit(1, test_size=limit_ratio)
        splits = list(splitter.split(data, labels))
        data = [data[n] for n in splits[0][1]]
        labels = [labels[n] for n in splits[0][1]]
        if targets is not None:
            targets = [targets[n] for n in splits[0][1]]

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
            if targets is not None:
                del targets[idx]
            # remove by index of index
            idx_idx = maxfreq_label_idx.index(idx)
            del maxfreq_label_idx[idx_idx]
        return data, labels, targets

    def limit_data_simple(self, num_limit, data, labels=None, targets=None):
        """Simple data limiter"""
        idxs = random.sample(list(range(len(data))), num_limit)
        data = [data[i] for i in idxs]
        if labels is not None:
            labels = np.asarray([labels[i] for i in idxs])
        if targets is not None:
            targets = [targets[i] for i in idxs]
        return data, labels, targets

    def data_limit_collections(self, num_limit, data, labels=None, targets=None, is_multilabel=False, handle_labelset=True):
        num_data = len(data)
        is_labelled = labels is not None
        error(f"Attempted to data-limit {num_data} items to {num_limit}", num_data < num_limit)
        if is_labelled and not is_multilabel:
            # use stratification
            data, labels, targets = self.limit_data_stratify(num_limit, data, labels, targets)
            info(f"Limited loaded data to {len(data)} items.")
            error("Inconsistent limiting in data.", len({num_limit, len(data), len(labels)}) != 1)
        else:
            warning("Resorting to non-stratified limiting")
            data, labels, targets = self.limit_data_simple(num_limit, data, labels, targets)
        labelset = None
        if is_labelled and handle_labelset:
            labelset = get_labelset(labels)
            labels = align_index(labels, labelset)
        return data, labels, targets, labelset


    def apply_data_limit(self, data, labels=None, multilabel=None, targets=None):
        lim = self.dlim
        lim = lim + [-1] if len(lim) == 1 else lim
        lim = [x if x >= 0 else None for x in lim]
        ltrain, ltest = lim
        train, test = data
        is_labelled = labels[0] is not None
        if is_labelled:
            train_labels, test_labels = labels
        else:
            train_labels, test_labels = None, None
        if targets[0]:
            train_targets, test_targets = targets
        else:
            train_targets, test_targets = None, None
        labelset = None
        if ltrain:
            if len(train) < ltrain:
                error("Attempted to data-limit {} train items to {}".format(len(train), ltrain))
            train, train_labels, train_targets, labelset = self.data_limit_collections(ltrain, train, train_labels, train_targets, multilabel)

        if ltest and test:
            error(f"Attempted to data-limit {len(test)} test items to {ltest}", len(test) < ltest)
            test, test_labels, test_targets, _ = self.data_limit_collections(ltest, test, test_labels, test_targets, multilabel, handle_labelset=False)
        targets = (train_targets, test_targets)
        if labels:
            return (train, test), (train_labels, test_labels), labelset, targets
        return (train, test), labels, labelset, targets

    def restrict_to_classes(self, data, labels, targets, restrict_classes):
        new_data, new_labels, new_targets = [], [], []
        # ensure iterables
        restore_ndarray = False
        if type(labels) is np.ndarray:
            labels = [[l] for l in labels]
            restore_ndarray = True
        for d, l, t in zip(data, labels, targets):
            valid_classes = [cl for cl in l if cl in restrict_classes]
            if not valid_classes:
                continue
            new_data.append(d)
            new_labels.append(valid_classes)
            new_targets.append(t)

        if restore_ndarray:
            new_labels = np.asarray([x for l in new_labels for x in l])
        return new_data, new_labels, new_targets

    def apply_class_limit(self, data, labels, labelset, label_names, multilabel, targets):
        c_lim = self.clim
        num_labels = len(labelset)
        error("Specified non-sensical class limit from {} classes to {}.".format(num_labels, c_lim), c_lim >= num_labels)
        if data:
            train, test = data
            train_labels, test_labels = labels
            train_targets, test_targets = targets
            # data have been loaded -- apply limit
            retained_classes = random.sample(labelset, c_lim)
            info("Limiting to the {}/{} classes: {}".format(c_lim, num_labels, retained_classes))
            if multilabel:
                debug("Max train/test labels per item prior: {} {}".format(max(map(len, train_labels)), max(map(len, test_labels))))
            train, train_labels = self.restrict_to_classes(train, train_labels, train_targets, retained_classes)
            test, test_labels = self.restrict_to_classes(test, test_labels, test_targets, retained_classes)
            num_labels = len(retained_classes)
            error("Zero labels after limiting.", not num_labels)
            # remap retained classes to indexes starting from 0
            train_labels = align_index(train_labels, retained_classes)
            test_labels = align_index(test_labels, retained_classes)
            # fix the label names
            label_names = [label_names[rc] for rc in retained_classes]
            labelset = list(range(len(retained_classes)))
            if multilabel:
                debug("Max train/test labels per item post: {} {}".format(max(map(len, train_labels)), max(map(len, test_labels))))
        train_labels = [np.asarray(x) for x in train_labels]
        test_labels = [np.asarray(x) for x in test_labels]
        return (train, test), (train_labels, test_labels), (labelset, label_names), (train_targets, test_targets)

    def subsample(self, data, labels=None, labelset=None, label_names=None, multilabel=None, targets=None):
        """Apply data and class sub-sampling"""
        if any(l is not None for l in labels) and self.clim is not None:
            data, labels, (labelset, label_names), targets = self.apply_class_limit(data, labels, labelset, label_names, multilabel, targets)

        if self.dlim is not None:
            data, labels, labelset, targets = self.apply_data_limit(data, labels, multilabel, targets)

        return data, labels, labelset, label_names,  targets

    # endregion

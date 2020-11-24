import numpy as np
from utils import error, info
"""Class that expands data samples"""

class DataAugmentation:

    transformation_function = None

    def __init__(self, func=None, collection_func=None):
        if func is None:
            self.transformation_function = lambda x: x
        self.transformation_function = func

    # applies the transformation function on the data
    def augment_single(self, data):
        return self.transformation_function(data)

    def augment_collection(self, data_collection, selection=None):
        if selection is None:
            # randomly select
            selection = [np.random.choice(range(len(data_collection)))]
        for idx in selection:
            new_datum = self.transformation_function(data_collection[idx])
            if type(data_collection) is list:
                data_collection.append(new_datum)
            elif type(data_collection) is np.ndarray:
                data_collection = np.append(data_collection, new_datum)
        return data_collection, selection


class LabelledDataAugmentation(DataAugmentation):

    def augment_single(self, data, label):
        return DataAugmentation.transformation_function(self, data, label)

    def augment_collection(self, data_collection, labels, selection=None):
        aug_data, _ = self.augment_collection(data_collection, selection)
        return aug_data, labels[selection]

    def oversample_to_ratio(self, data_collection, labels, labels_of_interest, desired_ratio, only_indexes=False, limit_to_indexes=None):
        l1, l2 = [np.where(labels == li)[0] for li in labels_of_interest]
        if limit_to_indexes is not None:
            l1, l2 = [li[np.where(np.intersect1d(li, limit_to_indexes))[0]] for li in [l1, l2]]

        new_samples = []
        ratio = len(l1) / len(l2)
        error("Ratio of labels {} is {} and cannot be modified to {} by oversampling the latter ({})".format(labels_of_interest, ratio, desired_ratio, labels_of_interest[-1]), ratio < desired_ratio)
        while ratio > desired_ratio:
            selected_sample = np.random.choice(l2)
            new_samples.append(selected_sample)
            if not only_indexes:
                # oversample a random  sample of second label
                data_collection, labels = self.augment_collection(data_collection, labels, selection=selected_sample)
            ratio = len(l1) / (len(l2) + len(new_samples))
        if only_indexes:
            return np.asarray(new_samples, np.int32)
        return data_collection, labels

    def undersample_to_ratio(self, data_collection, labels, label_indexes, desired_ratio, only_indexes):
        l1, l2 = [np.where(labels == i) for i in label_indexes]
        l1, l2 = l1[0], l2[0]
        new_samples = []
        ratio = len(l1) / len(l2)
        error("Ratio of labels {} is {} and cannot be modified to {} by oversampling the latter ({})".format(label_indexes, ratio, desired_ratio, label_indexes[-1]),
              ratio > desired_ratio)
        while ratio < desired_ratio:
            selected_sample = np.random.choice(l2)
            new_samples.append(selected_sample)
            if not only_indexes:
                # oversample a random  sample of second label
                data_collection, labels = np.delete(data_collection, selected_sample), np.delete(labels, selected_sample)
            ratio = len(l1) / (len(l2) - len(new_samples))
        if only_indexes:
            return np.asarray(new_samples, np.int32)
        return data_collection, labels

    def get_label_ratio(labels, label_indexes):
        label1, label2 = [labels[i] for i in label_indexes]
        return len(labels[labels == label1]) / len(labels[labels == label2])

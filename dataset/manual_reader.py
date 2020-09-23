"""Module for reading custom serialized datasets"""
import json
import numbers

import numpy as np

from utils import error


class ManualDatasetReader:
    train_labels, test_labels = None, None
    train_targets, test_targets = None, None
    train, test  = None, None
    labelset, label_names = None, None

    def handle_instance_labels(self, lbls, current_labelset, current_label_names):
        """Ensure iterable of numeric labels"""
        # iterable check
        try:
            lbls[0]
        except TypeError:
            lbls = [lbls]

        for i in range(len(lbls)):
            l = lbls[i]
            # numeric check
            if not isinstance(l, numbers.Number):
                if l not in current_label_names:
                    current_label_names.append(l)
                lbls[i] = current_label_names.index(l)
                current_labelset.add(lbls[i])
        return lbls, current_labelset
                

        
    """Reader class for reading custom serialized datasets"""
    def read_instances(self, json_object, data_key="text", labels_key="labels", targets_key="targets"):
        """Read a json object containing text dataset instances

        Arguments:
            json_object {dict} -- The json object containing instances
        Keyword arguments:
            data_key {str} -- The key under which the data is contained (default: "text")
            labels_key {str} -- The key under which the labels is contained (default: "labels")
            target_key{str} -- The key under which the generic targets are contained (e.g. textual) (default: "target")

        """
        data, labels, labelset, max_num_instance_labels, label_names = [], [], set(), 0, []
        targets = []

        is_labelled, is_fully_labelled = False, None
        has_ground_truth = False
        for instance in json_object:
            # read data
            data.append(instance[data_key])
            # read labels
            if labels_key in instance:
                is_labelled = True
                lbls = instance[labels_key]
                lbls, labelset = self.handle_instance_labels(lbls, labelset, label_names)
                labels.append(lbls)
                labelset.update(lbls)
                # check for multi-label setting
                if len(lbls) > max_num_instance_labels:
                    max_num_instance_labels = len(lbls)
            else:
                # mark the instance as unlabelled, if labels have
                # been read previously
                if is_labelled:
                    labels.append([])
                    is_fully_labelled = False
            # read other targets
            if targets_key in instance:
                has_ground_truth = True
                # since targets are non-numeric, keep the format as-is without inserting into a list
                # if type(instance[targets_key]) != list:
                #     instance[targets_key] = [instance[targets_key]]
                targets.append(instance[targets_key])

        return data, labels, labelset, label_names, is_labelled, is_fully_labelled, max_num_instance_labels, targets

    def read_json_dataset(self, path=None, data=None):
        """ Read a JSON-serialized dataset from a folder.

        Expected format in the yml config:
        name: path/to/dataset_name.json

        In the above path, define dataset json as:
        {
            data:
                train:
                    [
                        {
                            text: "this is the document text"
                            labels: [0,2,3]
                        },
                        ...
                    ],
                test: [...]
            num_labels: 10
            label_names: ['cat', 'dog', ...]
            language: english
        }
        """

        if path is not None:
            # read json file
            with open(path) as f:
                json_data = json.load(f)
        else:
            json_data = data

        data = json_data["data"]
        # read training data
        self.train, train_labels, train_labelset, train_label_names, self.train_is_labelled, self.train_fully_labelled, self.max_num_instance_labels,  self.train_targets = \
            self.read_instances(data["train"])

        try:
            self.test, test_labels, test_labelset, test_label_names, self.test_is_labelled, self.test_fully_labelled, _, self.test_targets = \
                self.read_instances(data["test"])
        except KeyError:
            pass

        self.roles = ("train", "test")
        # read metadata
        try:
            self.language = json_data["language"]
        except KeyError:
            # default to english
            self.language = "english"
        # process labelling
        if train_labelset:
            # labelsets
            # ensure training labelset is at least as large as the test one
            if self.test_is_labelled:
                ntrain, ntest = [len(x) for x in (train_labelset, test_labelset)]
                if ntrain < ntest:
                    error("Read manual dataset with {ntrain} unique labels in the training set, but {ntest} in the test set.")
            # collapse to single
            self.labelset = train_labelset
            # label names
            if "label_names" in json_data:
                # read manual label names
                self.label_names = json_data["label_names"]
                if type(self.label_names) is not list:
                    error(f"Expected list of strings for labelnames, got {type(self.label_names)} : {self.label_names}")
                if train_label_names and train_label_names != self.label_names:
                    error(f"Train non-numeric label names differ than the global labelnames provided.")

            elif train_label_names:
                self.label_names = train_label_names
            else:
                # assign numeric indexes as label names
                self.label_names = [str(x) for x in self.labelset]

            # always set labels to ndarray lists, for compatibility for multilabel data
            self.train_labels = [np.asarray(x) for x in train_labels]
            self.test_labels = [np.asarray(x) for x in test_labels]


    def read_dataset(self, raw_data, format="json"):
        """Read a manual dataset based on the configuration options"""
        # just JSON support for now
        if format == "json":
            self.read_json_dataset(data=raw_data)
            return
        error(f"Undefined custom dataset format: {format}")

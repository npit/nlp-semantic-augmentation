"""Module for reading custom serialized datasets"""
import json

import numpy as np

from utils import error


class ManualDatasetReader:
    """Reader class for reading custom serialized datasets"""
    def read_instances(self, json_object, data_key="text", labels_key="labels"):
        
        """Read a json object containing text dataset instances

        Arguments:
            json_object {dict} -- The json object containing instances
        Keyword arguments:
            data_key {str} -- The key under which the data is contained (default: "text")
            labels_key {str} -- The key under which the labels is contained (default: "labels")

        """
        data, labels, labelset, max_num_instance_labels = [], [], set(), 0
        is_labelled, is_fully_labelled = False, None
        for instance in json_object:
            # read data
            data.append(instance[data_key])
            # read labels
            if labels_key in instance:
                is_labelled = True
                lbls = instance[labels_key]
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

        return data, labels, labelset, is_labelled, is_fully_labelled, max_num_instance_labels

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
        self.train, train_labels, train_labelset, self.train_is_labelled, self.train_fully_labelled, self.max_num_instance_labels = \
            self.read_instances(data["train"])
        self.test, test_labels, test_labelset, self.test_is_labelled, self.test_fully_labelled, _ = \
            self.read_instances(data["test"])

        # read metadata
        try:
            self.language = json_data["language"]
        except KeyError:
            self.language = "UNDEFINED"
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
            else:
                # assign numeric indexes as label names
                self.label_names = [str(x) for x in self.labelset]
            if self.max_num_instance_labels > 1:
                # labels to ndarray lists
                self.train_labels = [np.asarray(x) for x in train_labels]
                self.test_labels = [np.asarray(x) for x in test_labels]
            else:
                # labels to ndarray
                self.train_labels = np.squeeze(np.asarray(train_labels, dtype=np.int32))
                self.test_labels = np.squeeze(np.asarray(test_labels, dtype=np.int32))

    def read_dataset(self, raw_data, format="json"):
        """Read a manual dataset based on the configuration options"""
        # just JSON support for now
        if format == "json":
            return self.read_json_dataset(data=raw_data)
        error(f"Undefined custom dataset format: {format}")

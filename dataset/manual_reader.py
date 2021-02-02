"""Module for reading custom serialized datasets"""
import json
import numbers
import defs
import numpy as np
from collections import OrderedDict

from utils import error, update_cumulative_index, debug


class ManualDatasetReader:
    label_names = None, None
    data, roles = None, None
    max_num_instance_labels = -1

    def handle_instance_labels(self, raw_lbls, current_label_names):
        """Ensure iterable of numeric labels"""
        # iterable check
        try:
            raw_lbls[0]
        except IndexError:
            raw_lbls = []
        except TypeError:
            raw_lbls = [raw_lbls]

        output_lbls = []
        for i in range(len(raw_lbls)):
            l = raw_lbls[i]
            # numeric check
            if not isinstance(l, numbers.Number):
                if l not in current_label_names:
                    current_label_names.append(l)
                # get numeric index of name
                output_lbls.append(current_label_names.index(l))
            else:
                # numeric, just append
                output_lbls.append(l)
        return output_lbls

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
        data, labels, max_num_instance_labels, label_names = [], [], 0, []
        targets = []

        is_labelled, is_fully_labelled = False, None

        for instance in json_object:
            # read data
            data.append(instance[data_key])
            # read labels
            if labels_key in instance:
                is_labelled = True
                raw_lbls = instance[labels_key]
                lbls = self.handle_instance_labels(raw_lbls, label_names)
                labels.append(lbls)

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
                targets.append(instance[targets_key])

        return data, labels, label_names, is_labelled, is_fully_labelled, max_num_instance_labels, targets

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


        # read metadata
        try:
            self.language = json_data["language"]
        except (TypeError, KeyError):
            # default to english
            self.language = "english"



        # read training data
        self.data, self.labels, self.targets, self.roles = [], [], [], []
        self.is_labelled = False
        self.indices = []

        for role in [defs.roles.train, defs.roles.test]:
            if role not in json_data['data']:
                continue
            data, labels, label_names, is_labelled, fully_labelled, max_num_instance_labels, targets = \
                self.read_instances(json_data['data'][role])

            offset = len(self.data)
            idx = np.arange(offset, len(data) + offset)
            self.indices.append(idx)
            self.data.extend(data)
            self.labels.extend([np.asarray(x) for x in labels])
            if targets:
                self.targets.extend(targets)
            self.roles.append(role)

            self.is_labelled = self.is_labelled or is_labelled
            self.max_num_instance_labels = max_num_instance_labels if self.max_num_instance_labels < max_num_instance_labels else self.max_num_instance_labels

            if role == defs.roles.train:
                self.label_names = label_names

        # process labelling
        if self.is_labelled:
            self.configure_labelnames(json_data, self.label_names)


    def configure_labelnames(self, json_data, instances_labelnames):
        # configure label names
        # label names
        if "label_names" in json_data:
            # read explicitely defined label names
            explicit_label_names = json_data["label_names"]

            if type(explicit_label_names) is not list:
                error(f"Expected list of strings for labelnames, got {type(explicit_label_names)} : {explicit_label_names}")

            if sorted(instances_labelnames) != explicit_label_names:
                error(f"Label names read from instances: {instances_labelnames} differ from ones explicitely defined: {explicit_label_names}")

        #     if instances_labelnames != self.label_names:
        #         # need instances with numeric labels
        #         debug(f"Train label names differ from the global labelnames provided -- mapping.")
        #         if type(instances_labelset[0]) == str:
        #             # better be the same labelset
        #             if instances_labelset != self.label_names:
        #                 error(f"Train string label names differ from the global string labelnames provided.")
        #             # remap to ints
        #             for i in range(len(self.labels)):
        #                 str_lbls = self.labels[i]
        #                 int_lbls = [self.label_names.index(l) for l in str_lbls]
        #                 self.labels[i] = np.split(np.asarray(int_lbls), len(int_lbls))
        #         else:
        #             if len(instances_labelset) != len(self.label_names):
        #                 error(f"Unequal labelset lengths from instances: {len(instances_labelset)} and dedicated field: {len(self.label_names)}")
        # else:
        #     # assign numeric indexes as label names
        #     self.label_names = [str(x) for x in self.labelset]


    def read_dataset(self, raw_data, format="json"):
        """Read a manual dataset based on the configuration options"""
        # just JSON support for now
        if format == "json":
            self.read_json_dataset(data=raw_data)
            return
        error(f"Undefined custom dataset format: {format}")


    @staticmethod
    def instances_to_json_dataset(instances, language="english", label_names=None):
        if type(instances) is str:
            instances = json.loads(instances)
        if type(instances) is list and type(instances[0]) is str:
            # instances = [{"text": inst, "labels": []} for inst in instances]
            instances = [{"text": inst} for inst in instances]
        dset = {"data":{"test": instances}, "language": language}
        if label_names is not None:
            dset["label_names"] = label_names
        return dset

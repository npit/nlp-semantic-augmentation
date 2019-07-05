import json
from os.path import basename
import numpy as np
from dataset.dataset import Dataset
from utils import write_pickled


class ManualDataset(Dataset):
    """ Class to import a dataset from a folder.

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

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = basename(config.name)
        Dataset.__init__(self)

    def get_all_raw(self):
        data = Dataset.get_all_raw(self)
        data["language"] = self.language
        data["multilabel"] = self.is_multilabel()
        return data

    # raw path getter
    def get_raw_path(self):
        return self.config.name

    def fetch_raw(self, raw_data_path):
        # no limited dataset
        if self.name != self.base_name:
            return None

        with open(raw_data_path) as f:
            raw_data = json.load(f)
        return raw_data

    def handle_raw(self, raw_data):
        max_num_instance_labels = 0
        self.num_labels = raw_data["num_labels"]
        self.language = raw_data["language"]
        data = raw_data["data"]

        self.train, self.train_labels = [], []
        self.test, self.test_labels = [], []

        unique_labels = {"train": set(), "test": set()}
        for obj in data["train"]:
            self.train.append(obj["text"])
            lbls = obj["labels"]
            self.train_labels.append(lbls)
            unique_labels["train"].update(lbls)
            max_num_instance_labels = len(lbls) if len(lbls) > max_num_instance_labels else max_num_instance_labels
        for obj in data["test"]:
            self.test.append(obj["text"])
            self.test_labels.append(obj["labels"])
            unique_labels["test"].update(obj["labels"])

        if "label_names" in raw_data:
            self.train_label_names = raw_data["label_names"]["train"]
            self.test_label_names = raw_data["label_names"]["test"]
        else:
            self.train_label_names, self.test_label_names = \
                [list(map(str, sorted(unique_labels[tt]))) for tt in ["train", "test"]]
        if max_num_instance_labels > 1:
            self.multilabel = True
        else:
            # labels to ndarray
            self.train_labels = np.squeeze(np.asarray(self.train_labels))
            self.test_labels = np.squeeze(np.asarray(self.test_labels))
        # write serialized data
        write_pickled(self.serialization_path, self.get_all_raw())

    def handle_raw_serialized(self, deserialized_data):
        Dataset.handle_raw_serialized(self, deserialized_data)
        self.language = deserialized_data["language"]
        self.multilabel = deserialized_data["multilabel"]

    def handle_serialized(self, deserialized_data):
        self.handle_raw_serialized(self, deserialized_data)

    def handle_preprocessed(self, deserialized_data):
        Dataset.handle_preprocessed(self, deserialized_data)

    def get_name(self):
        # get only the filename
        return basename(self.name)

    def get_base_name(self):
        # get only the base filename
        return basename(self.base_name)

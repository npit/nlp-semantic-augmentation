from nltk.corpus import reuters

from dataset.dataset import Dataset
from utils import info, nltk_download, warning, write_pickled


class Reuters(Dataset):
    name = "reuters"
    language = "english"

    def __init__(self, config):
        self.config = config
        self.multilabel = True
        self.base_name = self.name
        Dataset.__init__(self)

    def fetch_raw(self, dummy_input):
        # only applicable for raw dataset
        if self.name != self.base_name:
            return None
        info("Downloading raw {} dataset".format(self.name))
        if not self.nltk_dataset_resource_exists(Reuters.name):
            nltk_download(self.config, "reuters")
        # get ids
        categories = reuters.categories()
        self.num_labels = len(categories)
        self.label_names = []
        # train / test labels
        samples = {}
        train_docs, test_docs = [], []
        doc2labels = {}

        # get content
        for cat_index, cat in enumerate(categories):
            samples[cat] = [0, 0]

            # get all docs in that category
            for doc in reuters.fileids(cat):
                # document to label mappings
                if doc not in doc2labels:
                    # not encountered: init document label list
                    doc2labels[doc] = []
                    if doc.startswith("training"):
                        train_docs.append(doc)
                    else:
                        test_docs.append(doc)
                # count samples
                if doc.startswith("training"):
                    samples[cat][0] += 1
                else:
                    samples[cat][1] += 1
                # append the label
                doc2labels[doc].append(cat_index)

        doc2labels, label_set = self.delete_no_sample_labels(samples, doc2labels)

        self.train, self.test = [], []
        self.train_labels, self.test_labels = [], []
        # assign label lists
        for doc in train_docs:
            self.train.append(reuters.raw(doc))
            self.train_labels.append(doc2labels[doc])
        for doc in test_docs:
            self.test.append(reuters.raw(doc))
            self.test_labels.append(doc2labels[doc])

        self.label_names = label_set
        self.labelset = list(sorted(set(self.train_labels)))
        self.roles = "train", "test"
        info("Loaded {} train & {} test instances.".format(len(self.train), len(self.test)))
        return self.get_all_raw()

    # delete undersampled classes
    def delete_no_sample_labels(self, samples, doc2labels):
        # This reports different smaples: https://martin-thoma.com/nlp-reuters/
        labels2delete = []
        for label in samples:
            if any([x == 0 for x in samples[label]]):
                warning("Will remove label {} with samples: {}".format(label, samples[label]))
                labels2delete.append(label)
        if labels2delete:
            warning("Removing {} labels due to no train/test samples: {}".format(len(labels2delete), labels2delete))
            docs2delete = []
            for doc in doc2labels:
                new_labels = [l for l in doc2labels[doc] if l not in labels2delete]
                if not new_labels:
                    docs2delete.append(doc)
                doc2labels[doc] = new_labels
            for doc in docs2delete:
                del doc2labels[doc]
        return doc2labels, list(samples.keys())

    def handle_raw(self, raw_data):
        # serialize
        write_pickled(self.serialization_path, raw_data)
        self.loaded_raw = True
        pass

    # raw path getter
    def get_raw_path(self):
        # dataset is downloadable
        return None

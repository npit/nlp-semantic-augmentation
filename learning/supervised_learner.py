import numpy as np

from bundle.datatypes import Labels
from learning.learner import Learner
from utils import count_label_occurences, error, is_multilabel


class SupervisedLearner(Learner):
    train_labels, test_labels = None, None

    def count_samples(self):
        """Sample counter that includes the label samples"""
        super().count_samples()
        self.num_train_labels, self.num_test_labels = map(len, [self.train_labels, self.test_labels])

    def check_sanity(self):
        super().check_sanity()

        # need at least one sample per class
        zero_samples_idx = np.where(np.sum(self.train_labels, axis=0) == 0)
        if np.any(zero_samples_idx):
            error("No training samples for class index {}".format(
                zero_samples_idx))

        self.do_multilabel = is_multilabel(self.train_labels)
        label_counts = count_label_occurences(self.train_labels)
        self.num_labels = len(label_counts)


        # configure and sanity-check evaluator
        if self.validation_exists and not self.use_validation_for_training:
            # calculate the majority label from the training data -- label counts already computed
            self.evaluator.majority_label = label_counts[0][0]
            info("Majority label: {}".format(self.evaluator.majority_label))
            self.evaluator.configure(self.train_labels, self.num_labels,
                                     self.do_multilabel,
                                     self.use_validation_for_training,
                                     self.validation_exists)
            self.evaluator.compute_label_distribution()
        else:
            # count label distribution from majority
            self.evaluator.configure(self.test_labels, self.num_labels,
                                     self.do_multilabel,
                                     self.use_validation_for_training,
                                     self.validation_exists)
            self.evaluator.majority_label = count_label_occurences(
                self.test_labels)[0][0]


    def configure_validation_setting(self):
        self.validation = Learner.ValidatonSetting(self.folds,
                                                   self.validation_portion,
                                                   self.test_data_available(),
                                                   self.do_multilabel)
        self.validation.assign_data(self.embeddings, train_index=self.train_index, train_labels=self.train_labels, test_labels=self.test_labels, test_index=self.test_index)

    def configure_sampling(self):
        """Over/sub sampling"""
        if self.do_sampling:
            if type(self.sampling_ratios[0]) is not list:
                self.sampling_ratios = [self.sampling_ratios]
            freqs = count_label_occurences(
                [x for y in self.sampling_ratios for x in y[:2]])
            max_label_constraint_participation = max(freqs, key=lambda x: x[1])
            error(
                "Sampling should be applied on binary classification or constraining ratio should not be overlapping",
                self.num_labels > 2
                and max_label_constraint_participation[1] > 1)

    def show_train_statistics(self, train_labels, val_labels):
        self.evaluator.show_label_distribution(
            count_label_occurences(train_labels),
            "Training label distribution for validation setting: " +
            self.validation.get_current_descr())
        if val_labels is not None:
            self.evaluator.show_label_distribution(
                count_label_occurences(val_labels),
                "Validation label distribution for validation setting: "
                + self.validation.get_current_descr())

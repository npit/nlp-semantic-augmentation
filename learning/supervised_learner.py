import numpy as np

from bundle.datatypes import Labels
from learning.learner import Learner
from learning.sampler import Sampler
from utils import count_label_occurences, error, info, is_multilabel
from validation import ValidationSetting


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
        self.validation = ValidationSetting(self.folds,
                                                   self.validation_portion,
                                                   self.test_data_available(),
                                            use_labels=True,
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



    # produce training / validation splits, with respect to sample indexes
    def compute_trainval_indexes(self):
        if not self.validation_exists:
            # return all training indexes, no validation
            return [(np.arange(len(self.train_index)), np.arange(0))]

        trainval_serialization_file = self.get_trainval_serialization_file()

        if self.do_folds:
            # stratified fold splitting
            info(
                "Training {} with input data: {} samples, {} labels, on {} stratified folds"
                .format(self.name, self.num_train, self.num_train_labels,
                        self.folds))
            # for multilabel K-fold, stratification is not available. Also convert label format.
            if self.do_multilabel:
                splitter = KFold(self.folds,
                                 shuffle=True,
                                 random_state=self.seed)
            else:
                splitter = StratifiedKFold(self.folds,
                                           shuffle=True,
                                           random_state=self.seed)
                # convert to 2D array
                self.train_labels = np.squeeze(self.train_labels)

        if self.do_validate_portion:
            info(
                "Splitting {} with input data: {} samples, {} labels, on a {} validation portion"
                .form,
                random_state=self.seed)

        # generate. for multilabel K-fold, stratification is not usable
        splits = list(
            splitter.split(np.zeros(self.num_train_labels), self.train_labels))

        # do sampling processing
        if self.do_sampling:
            smpl = Sampler()
            splits = smpl.sample()

        # save and return the splitter splits
        makedirs(dirname(trainval_serialization_file), exist_ok=True)
        write_pickled(trainval_serialization_file, splits)
        return splits

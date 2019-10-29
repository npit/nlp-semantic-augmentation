import random
import tqdm
import numpy as np
from os import listdir
from os.path import basename

from bundle.datatypes import Text, Labels
from component.component import Component
from bundle.bundle import Bundle
from utils import error, tictoc, info, write_pickled, align_index, debug, warning, nltk_download, flatten
from defs import is_none, avail_preprocessing_items, avail_preprocessing_rules
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import nltk
from semantic.wordnet import Wordnet

from serializable import Serializable


class Dataset(Serializable):
    name = ""
    component_name = "dataset"
    vocabulary = set()
    vocabulary_index = []
    word_to_index = {}
    limited_name = ""
    dir_name = "datasets"
    undefined_word_index = None
    preprocessed = False
    train, test = None, None
    multilabel = False
    data_names = ["train-data", "train-labels", "train-label-names",
                  "test-data", "test-labels", "test_label-names"]
    preprocessed_data_names = ["vocabulary", "vocabulary_index", "undefined_word_index"]

    @staticmethod
    def generate_name(config):
        name = basename(config.dataset.name)
        if config.dataset.prepro is not None:
            name += "_" + config.dataset.prepro
        return name

    def nltk_dataset_resource_exists(self, name):
        try:
            if (name + ".zip") in listdir(nltk.data.find("corpora")):
                return True
        except:
            warning("Unable to probe nltk corpora at path {}".format(nltk.data.path))
        return False

    # dataset creation
    def __init__(self, skip_init=False):
        Component.__init__(self, produces=[Text.name])
        random.seed(self.config.misc.seed)
        if skip_init or self.config is None:
            return
        Serializable.__init__(self, self.dir_name)
        self.config.dataset.full_name = self.name

    def populate(self):
        self.set_serialization_params()
        # check for limited dataset
        self.acquire_data()
        if self.loaded():
            # downloaded successfully
            self.loaded_index = self.load_flags.index(True)
        else:
            # check for raw dataset. Suspend limit and setup paths
            self.suspend_limit()
            self.set_serialization_params()
            # exclude loading of pre-processed data
            self.data_paths = self.data_paths[1:]
            self.read_functions = self.read_functions[1:]
            self.handler_functions = self.handler_functions[1:]
            # get the data but do not preprocess
            self.acquire_data()
            if not self.loaded():
                error("Failed to load dataset")
            self.loaded_index = self.load_flags.index(True)
            # reapply the limit
            self.apply_limit()

        self.config.dataset.full_name = self.name
        info("Acquired dataset:{}".format(str(self)))
        # sanity checks
        # only numeric labels
        try:
            for labels in [self.train_labels, self.test_labels]:
                list(map(int, flatten(labels)))
        except ValueError as ve:
            error("Non-numeric label encountered: {}".format(ve))
        except TypeError as ve:
            warning("Non-collection labelitem encountered: {}".format(ve))
        # zero train / test
        # error("Problematic values loaded.", zero_length(self.train, self.test))

    def handle_preprocessed(self, data):
        # info("Loaded preprocessed {} dataset from {}.".format(self.name, self.serialization_path_preprocessed))
        self.handle_raw_serialized(data)
        self.vocabulary, self.vocabulary_index, self.undefined_word_index = [data[name] for name in
                                                                             self.preprocessed_data_names]
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
        info("Loaded preprocessed data: {} train, {} test, with {} labels".format(len(self.train), len(self.test),
                                                                                  self.num_labels))
        self.loaded_raw_serialized = False
        self.loaded_preprocessed = True

    def suspend_limit(self):
        self.name = Dataset.generate_name(self.config)

    def is_multilabel(self):
        return self.multilabel

    def get_raw_path(self):
        error("Need to override raw path datasea getter for {}".format(self.name))

    def fetch_raw(self, dummy_input):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def handle_raw_serialized(self, deserialized_data):
        self.train, self.train_labels, self.train_label_names, \
        self.test, self.test_labels, self.test_label_names = \
            [deserialized_data[n] for n in self.data_names]
        self.num_labels = len(set(self.train_label_names))
        self.loaded_raw_serialized = True

    # data getter
    def get_data(self):
        return self.train, self.test

    def get_labels(self):
        return self.train_labels, self.test_labels

    def get_num_labels(self):
        return self.num_labels

    # apply stratifed  limiting to the data wrt labels
    def limit_data_stratify(num_limit, data, labels):
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

    def limit_data_simple(num_limit, data, labels):
        idxs = random.sample(list(range(len(data))), num_limit)
        data = [data[i] for i in idxs]
        labels = [labels[i] for i in idxs]
        return data, np.asarray(labels)

    def apply_data_limit(self, name):
        lim = self.config.dataset.data_limit
        lim = lim + [-1] if len(lim) == 1 else lim
        lim = [x if x >= 0 else None for x in lim]
        ltrain, ltest = lim
        if ltrain:
            name += "_dlim_tr{}".format(ltrain)
            if self.train:
                if len(self.train) < ltrain:
                    error("Attempted to data-limit {} train items to {}".format(len(self.train), ltrain))
                try:
                    # use stratification
                    self.train, self.train_labels = Dataset.limit_data_stratify(ltrain, self.train, self.train_labels)
                    info("Limited {} loaded data to {} train items.".format(self.base_name, len(self.train)))
                except ValueError as ve:
                    warning(ve)
                    warning("Resorting to non-stratified limiting")
                    self.train, self.train_labels = Dataset.limit_data_simple(ltrain, self.train, self.train_labels)
                if not len({ltrain, len(self.train), len(self.train_labels)}) == 1:
                    error("Inconsistent limiting in train data.")
        if ltest:
            name += "_dlim_te{}".format(ltest)
            if self.test:
                if len(self.test) < ltest:
                    error("Attempted to data-limit {} test items to {}".format(len(self.test), ltest))
                try:
                    # use stratification
                    self.test, self.test_labels = Dataset.limit_data_stratify(ltest, self.test, self.test_labels)
                    info("Limited {} loaded data to {} test items.".format(self.base_name, len(self.test)))
                except ValueError as ve:
                    warning(ve)
                    warning("Resorting to non-stratified limiting")
                    self.test, self.test_labels = Dataset.limit_data_simple(ltest, self.test, self.test_labels)
                if not len({ltest, len(self.test), len(self.test_labels)}) == 1:
                    error("Inconsistent limiting in test data.")
        return name

    def restrict_to_classes(self, data, labels, restrict_classes):
        new_data, new_labels = [], []
        for d, l in zip(data, labels):
            valid_classes = [cl for cl in l if cl in restrict_classes]
            if not valid_classes:
                continue
            new_data.append(d)
            new_labels.append(valid_classes)
        return new_data, new_labels

    def apply_class_limit(self, name):
        c_lim = self.config.dataset.class_limit
        if c_lim is not None:
            name += "_clim_{}".format(c_lim)
            if self.train:
                if c_lim >= self.num_labels:
                    error("Specified non-sensical class limit from {} classes to {}.".format(self.num_labels, c_lim))
                # data have been loaded -- apply limit
                retained_classes = random.sample(list(range(self.num_labels)), c_lim)
                info("Limiting to the {}/{} classes: {}".format(c_lim, self.num_labels, retained_classes))
                if self.multilabel:
                    debug("Max train/test labels per item prior: {} {}".format(max(map(len, self.train_labels)),
                                                                               max(map(len, self.test_labels))))
                self.train, self.train_labels = self.restrict_to_classes(self.train, self.train_labels,
                                                                         retained_classes)
                self.test, self.test_labels = self.restrict_to_classes(self.test, self.test_labels, retained_classes)
                self.num_labels = len(retained_classes)
                if not self.num_labels:
                    error("Zero labels after limiting.")
                # remap retained classes to indexes starting from 0
                self.train_labels = align_index(self.train_labels, retained_classes)
                self.test_labels = align_index(self.test_labels, retained_classes)
                # fix the label names
                self.train_label_names = [self.train_label_names[rc] for rc in retained_classes]
                self.test_label_names = [self.test_label_names[rc] for rc in retained_classes]
                info("Limited {} dataset to {} classes: {} - i.e. {} - resulting in {} train and {} test data."
                     .format(self.base_name, self.num_labels, retained_classes,
                             self.train_label_names, len(self.train), len(self.test)))
                if self.multilabel:
                    debug("Max train/test labels per item post: {} {}".format(max(map(len, self.train_labels)),
                                                                              max(map(len, self.test_labels))))
        return name

    def apply_limit(self):
        if self.config.has_limit():
            self.base_name = Dataset.generate_name(self.config)
            name = self.base_name
            if not is_none(self.config.dataset.class_limit):
                name = self.apply_class_limit(self.base_name)
            if not is_none(self.config.dataset.data_limit):
                name = self.apply_data_limit(name)
            self.name = name
            self.set_serialization_params()
        if self.train:
            # serialize the limited version
            write_pickled(self.serialization_path, self.get_all_raw())

    def setup_nltk_resources(self):
        try:
            stopwords.words(self.language)
        except LookupError:
            nltk_download(self.config, "stopwords")

        self.stopwords = set(stopwords.words(self.language))
        try:
            nltk.pos_tag("Text")
        except LookupError:
            nltk_download(self.config, 'averaged_perceptron_tagger')
        try:
            [x("the quick brown. fox! jumping-over lazy, dog.") for x in [word_tokenize, sent_tokenize]]
        except LookupError:
            nltk_download(self.config, "punkt")

        # setup word prepro
        while True:
            try:
                if self.config.dataset.prepro == "stem":
                    self.stemmer = PorterStemmer()
                    self.word_prepro = lambda w_pos: (self.stemmer.stem(w_pos[0]), w_pos[1])
                elif self.config.dataset.prepro == "lemma":
                    self.lemmatizer = WordNetLemmatizer()
                    self.word_prepro = self.apply_lemmatizer
                else:
                    self.word_prepro = lambda x: x
                break
            except LookupError as err:
                error(err)
        # punctuation
        self.punctuation_remover = str.maketrans('', '', string.punctuation)
        self.digit_remover = str.maketrans('', '', string.digits)

    def apply_lemmatizer(self, w_pos):
        wordnet_pos = Wordnet.get_wordnet_pos(w_pos[1])
        if not wordnet_pos:
            return self.lemmatizer.lemmatize(w_pos[0]), w_pos[1]
        else:
            return self.lemmatizer.lemmatize(w_pos[0], wordnet_pos), w_pos[1]

    # map text string into list of stopword-filtered words and POS tags
    def process_single_text(self, text, punctuation_remover, digit_remover, word_prepro, stopwords):
        # debug("Processing raw text:\n[[[{}]]]".format(text))
        if self.config.dataset.pre_segm:
            text = self.apply_preprocessing_rules(text=text, rules=self.config.dataset.pre_segm)
        sents = sent_tokenize(text.lower())
        words = []

        if self.config.dataset.post_segm:
            text = self.apply_preprocessing_rules(text=text, rules=self.config.dataset.post_segm)

        for sent in sents:
            # remove punctuation content
            sent = sent.translate(punctuation_remover)
            words.extend(word_tokenize(sent))
        # words = text_to_word_sequence(text, filters=filt, lower=True, split=' ')
        # words = [w.lower() for w in self.nltk_tokenizer.tokenize(text)]

        # remove stopwords and numbers
        # words = [w.translate(digit_remover) for w in words if w not in stopwords and w.isalpha()]
        words = [w for w in [w.translate(digit_remover) for w in words if w not in stopwords] if w]
        # pos tagging
        words_with_pos = nltk.pos_tag(words)
        # stemming / lemmatization
        words_with_pos = [word_prepro(wp) for wp in words_with_pos]
        if not words_with_pos:
            warning("Text preprocessed to an empty list:\n{}".format(text))
            return None
        return words_with_pos

    def apply_preprocessing_rules(self, text, rules):
        for rule, action_item in rules.items():
            error("Rule {} not defined. Available are {}".format(rule, avail_preprocessing_rules),
                  rule not in avail_preprocessing_rules)
            replacing_token = ""
            if rule == "delete":
                replacing_token = " "
            elif rule == "replace":
                replacing_token = " "
            error("Some preprocessing items {} not defined. Available are {}".format(action_item, avail_preprocessing_items),
                  any(item not in avail_preprocessing_items for item in action_item))
            func = lambda x: text.replace(x, replacing_token)
            for item in action_item:
                if item == "hashtag":
                    text = func("#[^\\s]+")
                elif item == "mention":
                    text = func("@[^\\s]+ ")
                elif item == "retweets":
                    text = func(" RT ")
                    text = func("^RT ")
                elif item == "url":
                    text = func("www[^\\s]+")
                    text = func("http[^\\s]+")
                    text = func("https.[^\\s]+")
        return text

    # preprocess single
    def preprocess_text_collection(self, document_list, track_vocabulary=False):
        # filt = '!"#$%&()*+,-./:;<=>?@\[\]^_`{|}~\n\t1234567890'
        ret_words_pos, ret_voc = [], set()
        num_words = []
        with tqdm.tqdm(desc="Mapping document collection", total=len(document_list), ascii=True, ncols=100,
                       unit="collection") as pbar:
            for i in range(len(document_list)):
                pbar.set_description("Document {}/{}".format(i + 1, len(document_list)))
                pbar.update()
                # text_words_pos = self.process_single_text(document_list[i], filt=filt, stopwords=stopw)
                text_words_pos = self.process_single_text(document_list[i],
                                                          punctuation_remover=self.punctuation_remover,
                                                          digit_remover=self.digit_remover,
                                                          word_prepro=self.word_prepro, stopwords=self.stopwords)
                if text_words_pos is None:
                    error("Text {}/{} preprocessed to an empty list:\n{}".format(i + 1, len(document_list),
                                                                                 document_list[i]))

                ret_words_pos.append(text_words_pos)
                if track_vocabulary:
                    ret_voc.update([wp[0] for wp in text_words_pos])
                num_words.append(len(text_words_pos))
        stats = [x(num_words) for x in [np.mean, np.var, np.std]]
        info("Words per document stats: mean {:.3f}, var {:.3f}, std {:.3f}".format(*stats))
        return ret_words_pos, ret_voc

    # preprocess raw texts into word list
    def preprocess(self):
        if self.loaded_preprocessed:
            info("Skipping preprocessing, preprocessed data already loaded from {}.".format(
                self.serialization_path_preprocessed))
            return
        self.setup_nltk_resources()
        with tictoc("Preprocessing {}".format(self.name)):
            info("Mapping training set to word collections.")
            self.train, self.vocabulary = self.preprocess_text_collection(self.train, track_vocabulary=True)
            info("Mapping test set to word collections.")
            self.test, _ = self.preprocess_text_collection(self.test)
            # fix word order and get word indexes
            self.vocabulary = list(self.vocabulary)
            for index, word in enumerate(self.vocabulary):
                self.word_to_index[word] = index
                self.vocabulary_index.append(index)
            # add another for the missing word
            self.undefined_word_index = len(self.vocabulary)
        # serialize
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_all_raw(self):
        return {"train-data": self.train, "train-labels": self.train_labels,
                "train-label-names": self.train_label_names,
                "test-data": self.test, "test-labels": self.test_labels, "test_label-names": self.test_label_names}

    def get_all_preprocessed(self):
        res = self.get_all_raw()
        res['vocabulary'] = self.vocabulary
        res['vocabulary_index'] = self.vocabulary_index
        res['undefined_word_index'] = self.undefined_word_index
        return res

    def get_name(self):
        return self.name

    def get_word_lists(self):
        """Get word-only data"""
        res = [], []
        for doc in self.train:
            res[0].append([wp[0] for wp in doc])
        for doc in self.test:
            res[1].append([wp[0] for wp in doc])
        return res

    def __str__(self):
        try:
            return "{}, train/test {}/{}, num labels: {}".format(self.base_name, len(self.train), len(self.test),
                                                                 self.num_labels)
        except:
            return self.base_name

    # region # chain methods

    def load_inputs(self, inputs):
        error("Attempted to load inputs into a {} component.".format(self.base_name), inputs is not None)

    def configure_name(self):
        self.apply_limit()
        Component.configure_name(self)

    def run(self):
        self.populate()
        self.preprocess()
        self.outputs.set_text(Text((self.train, self.test), self.vocabulary))
        self.outputs.set_labels(Labels((self.train_labels, self.test_labels)))
    # endregion

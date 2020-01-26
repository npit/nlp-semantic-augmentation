import random
import string
from os import listdir
from os.path import basename

import nltk
import numpy as np
import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from bundle.datatypes import Labels, Text
from component.component import Component
from dataset.sampling import Sampler
from semantic.wordnet import Wordnet
from serializable import Serializable
from utils import (error, flatten, info, nltk_download, tictoc, warning,
                   write_pickled)


class Dataset(Serializable):
    """Generic class for datasets."""
    name = ""
    component_name = "dataset"
    vocabulary = set()
    vocabulary_index = []
    word_to_index = {}
    dir_name = "datasets"
    undefined_word_index = None
    preprocessed = False
    train, test = None, None
    data_names = ["train-data", "test-data"]
    label_data_names = ["train-labels", "label-names", "test-labels"]
    train_labels, test_labels = None, None
    multilabel = False

    preprocessed_data_names = ["vocabulary", "vocabulary_index", "undefined_word_index"]

    @staticmethod
    def generate_name(config):
        """Generate a dataset identifier name

        Arguments:
            config {namedtuple} -- The configuration object
        Returns:
            The generated name
        """

        name = basename(config.dataset.name)
        if config.dataset.prepro is not None:
            name += "_" + config.dataset.prepro
        return name

    def nltk_dataset_resource_exists(self, name):
        """Checks the availability of NLTK resources

        Arguments:
            name {str} -- The name of the resource to check
        Returns:
            A boolean denoting existence of the resource.
        """

        try:
            if (name + ".zip") in listdir(nltk.data.find("corpora")):
                return True
        except:
            warning("Unable to probe nltk corpora at path {}".format(nltk.data.path))
        return False

    def __init__(self, skip_init=False):
        """Dataset constructor

        Arguments:
            skip_init {bool} -- Wether to initialize the serializable superclass mechanism
        """
        Component.__init__(self, produces=[Text.name])
        random.seed(self.config.misc.seed)
        if skip_init or self.config is None:
            return
        Serializable.__init__(self, self.dir_name)
        self.config.dataset.full_name = self.name

    def populate(self):
        self.set_serialization_params()
        self.acquire_data()
        if self.loaded():
            # downloaded successfully
            self.loaded_index = self.load_flags.index(True)
        else:
            # if the dataset's limited, check for the full version, else fail
            error("Failed to acquire dataset", not self.config.has_limit())
            # check for raw dataset. Suspend limit and setup paths
            self.name = Dataset.generate_name(self.config)
            self.set_serialization_params()
            # exclude loading of pre-processed data
            self.data_paths = self.data_paths[1:]
            self.read_functions = self.read_functions[1:]
            self.handler_functions = self.handler_functions[1:]
            # get the data but do not preprocess
            self.acquire_data()
            error("Failed to load dataset", not self.loaded())
            self.loaded_index = self.load_flags.index(True)
            # reapply the limit
            data, labels, self.labelset, self.label_names = self.sampler.subsample(
                self.get_data(), self.get_labels(), self.labelset, self.label_names, self.multilabel)
            self.train, self.test = data
            self.train_labels, self.test_labels = labels
            self.name = self.sampler.get_limited_name(self.name)
            self.set_serialization_params()
            write_pickled(self.serialization_path, self.get_all_raw())

        self.config.dataset.full_name = self.name
        info("Acquired dataset:{}".format(str(self)))
        self.check_sanity()

    def check_sanity(self):
        """Placeholder class for sanity checking"""
        # ensure numeric labels
        try:
            for labels in [self.train_labels, self.test_labels]:
                list(map(int, flatten(labels)))
        except ValueError as ve:
            error("Non-numeric label encountered: {}".format(ve))
        except TypeError as ve:
            warning("Non-collection labelitem encountered: {}".format(ve))

    def handle_preprocessed(self, data):
        # info("Loaded preprocessed {} dataset from {}.".format(self.name, self.serialization_path_preprocessed))
        self.handle_raw_serialized(data)
        self.vocabulary, self.vocabulary_index, self.undefined_word_index = [data[name] for name in self.preprocessed_data_names]
        for index, word in enumerate(self.vocabulary):
            self.word_to_index[word] = index
        info(self.get_info())
        self.loaded_raw_serialized = False
        self.loaded_preprocessed = True

    #region getters
    def get_data(self):
        return self.train, self.test

    def is_multilabel(self):
        """Multilabel status getter"""
        return self.multilabel

    def get_labels(self):
        """Labels getter"""
        return self.train_labels, self.test_labels

    def get_num_labels(self):
        """Number of labels getter"""
        return self.num_labels

    def get_info(self):
        """Current data information getter"""
        return f"{self.name} data: {len(self.train)} train, {len(self.test)} test, {self.num_labels}"

    def get_raw_path(self):
        error("Need to override raw path datasea getter for {}".format(self.name))

    def fetch_raw(self, dummy_input):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def handle_raw_serialized(self, deserialized_data):
        self.train, self.test = [deserialized_data[n] for n in self.data_names]
        self.train_labels, self.label_names, self.test_labels = [deserialized_data[n] for n in self.label_data_names]
        self.num_labels = len(set(self.label_names))
        if type(self.train_labels) is np.ndarray:
            self.multilabel = False
        else:
            if type(self.train_labels[0]) is int:
                self.multilabel = False
            else:
                self.multilabel = any(len(ll) for ll in self.train_labels)
        self.loaded_raw_serialized = True

    def get_num_labels(self):
        """Placeholder number of labels getter"""
        return None
    #endregion

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
        sents = sent_tokenize(text.lower())
        words = []
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

    # preprocess single
    def preprocess_text_collection(self, document_list, track_vocabulary=False):
        # filt = '!"#$%&()*+,-./:;<=>?@\[\]^_`{|}~\n\t1234567890'
        ret_words_pos, ret_voc = [], set()
        num_words = []
        if not document_list:
            info("(Empty collection)")
            return [], []
        with tqdm.tqdm(desc="Mapping document collection", total=len(document_list), ascii=True, ncols=100, unit="collection") as pbar:
            for i in range(len(document_list)):
                pbar.set_description("Document {}/{}".format(i + 1, len(document_list)))
                pbar.update()
                # text_words_pos = self.process_single_text(document_list[i], filt=filt, stopwords=stopw)
                text_words_pos = self.process_single_text(document_list[i], punctuation_remover=self.punctuation_remover, digit_remover=self.digit_remover,
                                                          word_prepro=self.word_prepro, stopwords=self.stopwords)
                if text_words_pos is None:
                    error("Text {}/{} preprocessed to an empty list:\n{}".format(i+1,len(document_list), document_list[i]))

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
            info("Skipping preprocessing, preprocessed data already loaded from {}.".format(self.serialization_path_preprocessed))
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
        data = {"train-data": self.train, "test-data": self.test}
        for key, value in zip(self.label_data_names, [self.train_labels, self.label_names, self.test_labels]):
            data[key] = value
        return data

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
            return self.get_info()
        except:
            return self.base_name


    # region # chain methods

    def set_outputs(self):
        """Set text data to the output bundle"""
        self.outputs.set_text(Text((self.train, self.test), self.vocabulary))
        self.outputs.set_labels(Labels((self.train_labels, self.test_labels), self.multilabel))

    def load_inputs(self, inputs):
        error("Attempted to load inputs into a {} component.".format(self.base_name), inputs is not None)

    def configure_name(self):
        self.name = Dataset.generate_name(self.config)
        if self.config.has_limit():
            self.sampler = Sampler(self.config)
            self.name = self.sampler.get_limited_name(self.name)
        else:
            self.name = Dataset.generate_name(self.config)
        Component.configure_name(self)

    def run(self):
        self.populate()
        self.preprocess()
        self.set_outputs()
    # endregion

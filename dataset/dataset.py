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

from bundle.datatypes import *
from bundle.datausages import *
from component.component import Component
from dataset.sampling import Sampler
from semantic.wordnet import Wordnet
from serializable import Serializable
from utils import (error, flatten, info, nltk_download, tictoc, warning,
                   write_pickled, set_constant_epi)


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

    data, roles, indices = None, None, None
    labels, labelset, multilabel = None, None, None
    targets = None
    num_labels = None

    data_names = ["data", "indices", "roles"]
    target_data_names = ["targets"]
    label_data_names = ["labels", "label_names", "labelset"]

    preprocessed_data_names = ["vocabulary", "vocabulary_index", "undefined_word_index"]

    filter_stopwords = True
    stopwords = None

    produces = Text.name

    @staticmethod
    def generate_name(config):
        """Generate a dataset identifier name

        Arguments:
            config {namedtuple} -- The configuration object
        Returns:
            The generated name
        """

        name = basename(config.name)
        if config.prepro is not None:
            name += "_" + config.prepro
        if config.extract_pos:
            name += "_pos"
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
        random.seed(self.config.misc.seed)
        if skip_init or self.config is None:
            return
        Serializable.__init__(self, self.dir_name)
        self.config.full_name = self.name
        self.filter_stopwords = self.config.filter_stopwords
        self.remove_digits = self.config.remove_digits

    def load_model_from_disk(self):
        # for datasets, equivalent to loading the dataset
        self.acquire_data()
        return any(self.load_flags)

    def load_outputs_from_disk(self):
        # for datasets, equivalent to loading the preprocessed dataset
        return self.attempt_load(0)

    # def load_outputs_from_disk(self):
    #     # set serialization params here, after name's been configured
    #     import ipdb; ipdb.set_trace()
    #     self.acquire_data()
    #     if self.loaded() and (not self.config.has_limit() or self.sampler.matching_limits(self.indices, self.labelset)):
    #         # downloaded successfully
    #         self.loaded_index = self.load_flags.index(True)
    #     else:
    #         # if the dataset's limited, check for the full version, else fail
    #         if not self.config.has_limit():
    #             info(f"Dataset {self.name} not pre-defined and provided paths are not loadable/exist:{self.data_paths}")
    #             error(f"Failed to acquire {self.name} dataset")
    #         # check for raw dataset. Suspend limit and setup paths
    #         self.name = Dataset.generate_name(self.config)
    #         self.set_serialization_params()
    #         # exclude loading of pre-processed data
    #         self.data_paths = self.data_paths[1:]
    #         self.read_functions = self.read_functions[1:]
    #         self.handler_functions = self.handler_functions[1:]
    #         # get the data but do not preprocess
    #         self.acquire_data()
    #         error("Failed to load dataset", not self.loaded())
    #         self.loaded_index = self.load_flags.index(True)
    #         # reapply the limit
    #         train, test = self.indices.get_train_test()
    #         self.data, self.indices, self.labels, self.labelset, self.label_names, self.targets = self.sampler.subsample(
    #             self.data, (train, test), self.labels, self.labelset, self.label_names, self.multilabel, self.targets)
    #         self.name = self.sampler.get_limited_name(self.name)
    #         self.set_serialization_params()
    #         write_pickled(self.serialization_path, self.get_all_raw())
    #         self.loaded_preprocessed = False

    #     self.config.full_name = self.name
    #     info("Acquired dataset:{}".format(str(self)))
    #     self.check_sanity()
    #     return self.loaded_preprocessed

    def check_sanity(self):
        """Placeholder class for sanity checking"""
        # ensure numeric labels
        try:
            list(map(int, flatten(self.labels[:1])))
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
        self.indices = Indices(self.indices, roles=self.roles)
        self.language = data['language']

    # region getters
    def get_data(self):
        return self.data

    def is_multilabel(self):
        """Multilabel status getter"""
        return self.multilabel

    def get_targets(self):
        """Targets getter"""
        return self.targets

    def get_labels(self):
        """Labels getter"""
        return self.labels

    def is_labelled(self):
        """Boolean for labelling"""
        return self.labels is not None and self.labels[0] is not None

    def get_num_labels(self):
        """Number of labels getter"""
        return self.num_labels

    def get_info(self):
        """Current data information getter"""
        return f"{self.name} data: {len(self.data)}, {self.num_labels} labels, {len(self.targets) if self.targets is not None else None} targets"

    def get_raw_path(self):
        error("Need to override raw path datasea getter for {}".format(self.name))

    def fetch_raw(self, dummy_input):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def contains_multilabel(self, labels):
        """Checks wether labels contain multi-label annotations"""
        try:
            for instance_labels in labels:
                if len(instance_labels) > 1:
                    return True
        except TypeError:
            pass
        return False

    @staticmethod
    def get_text(data):
        """Get text from the dataset outputs"""
        return " ".join([item["words"] for item in data])

    def handle_raw_serialized(self, deserialized_data):
        self.data, self.indices, self.roles = [deserialized_data[n] for n in self.data_names]
        self.loaded_raw_serialized = True
        self.labels, self.label_names, self.labelset = [deserialized_data[n] for n in self.label_data_names]
        self.targets = deserialized_data[self.target_data_names[0]]
        if self.labels is not None:
            self.num_labels = len(set(self.label_names))
            self.multilabel = self.contains_multilabel(self.labels)

    # endregion

    def setup_nltk_resources(self):
        if self.filter_stopwords:
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

        # setup word preprocessing
        self.word_prepro_func = lambda x: x
        while True:
            try:
                if self.config.prepro == "stem":
                    error("Specified stemming without POS extraction.", not self.config.extract_pos)
                    self.stemmer = PorterStemmer()
                    self.word_prepro_func = lambda w_pos: (self.stemmer.stem(w_pos[0]), w_pos[1])
                elif self.config.prepro == "lemma":
                    error("Specified lemmatization without POS extraction.", not self.config.extract_pos)
                    self.lemmatizer = WordNetLemmatizer()
                    self.word_prepro_func = self.apply_lemmatizer
                else:
                    error(f"Undefined preprocessing opt: {self.config.prepro}", self.config.prepro is not None)
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

    def process_single_text(self, text, punctuation_remover, digit_remover, word_prepro_func, stopwords):
        """Apply processing for a single text element"""
        data = {}
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
        # remove empty "words"
        words = [w for w in words if w]
        if self.remove_digits:
            words = [w for w in [w.translate(digit_remover) for w in words] if w]
        if self.filter_stopwords:
            words = [w for w in words if w not in stopwords]
        data["words"] = words
        if self.config.extract_pos:
            # pos tagging
            data["pos"] = nltk.pos_tag(words)
        # stemming / lemmatization
        if self.config.prepro is not None:
            data["words"] = [word_prepro_func((w, p)) for (w, p) in zip(words, data["pos"])]
        # if not data["words"]:
        #     # warning("Text preprocessed to an empty list:\n{}".format(text))
        #     return None
        return data

    def has_text_targets(self):
        return self.targets is not None and len(self.targets) > 0 and type(self.targets[0]) == str

    # preprocess single
    def preprocess_text_collection(self, texts_container, container_idxs, track_vocabulary=False):
        # filt = '!"#$%&()*+,-./:;<=>?@\[\]^_`{|}~\n\t1234567890'
        ret_words_pos, ret_voc = [], set()
        num_words = []
        discarded_indexes = []
        if container_idxs.size == 0:
            info("(Empty collection)")
            return [], [], None
        with tqdm.tqdm(desc="Mapping document collection", total=len(container_idxs), ascii=True, ncols=100, unit="collection") as pbar:
            for i, idx in enumerate(container_idxs):
                data = texts_container[idx]
                pbar.set_description("Document {}/{}".format(i + 1, len(container_idxs)))
                pbar.update()
                data = self.process_single_text(data, punctuation_remover=self.punctuation_remover, digit_remover=self.digit_remover,
                                                          word_prepro_func=self.word_prepro_func, stopwords=self.stopwords)
                word_data = data["words"]
                if not word_data:
                    # warning("Text {}/{} preprocessed to an empty list:\n{}".format(i + 1, len(document_list), document_list[i]))
                    discarded_indexes.append(i)
                    # continue
                    data["words"] = []

                ret_words_pos.append(data)
                if track_vocabulary:
                    ret_voc.update(word_data)
                num_words.append(len(word_data))
        stats = [x(num_words) for x in [np.mean, np.var, np.std]]
        info("Words per document stats: mean {:.3f}, var {:.3f}, std {:.3f}".format(*stats))
        return ret_words_pos, ret_voc, discarded_indexes

    # preprocess raw texts into word list
    def produce_outputs(self):

        self.setup_nltk_resources()
        # make indices object -- this filters down non-existent (with no instances) roles
        self.indices = Indices(self.indices, roles=self.roles)
        self.roles = self.indices.roles
        train_idx, test_idx = self.indices.get_train_test()
        test_idx = self.indices.get_role_instances(defs.roles.test, must_exist=False)
        error("Neither train or test indices found to process dataset", not (train_idx.size > 0 or test_idx.size > 0))
        preproc_data = []
        preproc_targets = []

        with tictoc("Preprocessing {}".format(self.name)):
            info("Mapping text training data to word collections.")
            txts, self.vocabulary, discarded_indexes = self.preprocess_text_collection(self.data, train_idx, track_vocabulary=True)
            self.vocabulary = set(self.vocabulary)
            preproc_data.extend(txts)

            if self.has_text_targets():
                info("Mapping text training targets to word collections.")
                txts, voc, _ = self.preprocess_text_collection(self.targets, train_idx, track_vocabulary=True)
                self.vocabulary.update(voc)
                preproc_targets.extend(txts)

            # if discarded_indexes:
            #     warning(f"Discarded {len(discarded_indexes)} instances from preprocessing.")
            #     if self.train_labels is not None:
            #         self.train_labels = [self.train_labels[i] for i in range(len(self.train_labels)) if i not in discarded_indexes]

            info("Mapping text test data to word collections.")
            txts, _, discarded_indexes = self.preprocess_text_collection(self.data, test_idx)
            preproc_data.extend(txts)
            if self.has_text_targets():
                info("Mapping text test targets to word collections.")
                txts, _, _ = self.preprocess_text_collection(self.targets, test_idx)
                preproc_targets.extend(txts)

            # if discarded_indexes:
            #     warning(f"Discarded {len(discarded_indexes)} instances from preprocessing.")
            #     if self.test_labels is not None:
            #         self.test_labels = [self.test_labels[i] for i in discarded_indexes]
            # fix word order and get word indexes
            self.vocabulary = list(self.vocabulary)
            for index, word in enumerate(self.vocabulary):
                self.word_to_index[word] = index
                self.vocabulary_index.append(index)
            # add another for the missing word
            self.undefined_word_index = len(self.vocabulary)

            self.data = preproc_data
            self.targets = preproc_targets

    def save_outputs(self):
        # serialize preprocessed
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_all_raw(self):
        indices = self.indices.instances if type(self.indices) is Indices else self.indices
        data = {lbl: data for (lbl, data) in zip(self.data_names, (self.data, indices, self.roles))}
        for key, value in zip(self.label_data_names, self.get_labelled_data()):
            data[key] = value
        data[self.target_data_names[0]] = self.targets
        return data

    def get_labelled_data(self):
        return [self.labels, self.label_names, self.labelset]

    def get_all_preprocessed(self):
        res = self.get_all_raw()
        res['vocabulary'] = self.vocabulary
        res['vocabulary_index'] = self.vocabulary_index
        res['undefined_word_index'] = self.undefined_word_index
        return res

    def get_name(self):
        return self.name

    def __str__(self):
        try:
            return self.get_info()
        except:
            return self.base_name

    # region # chain methods

    def set_component_outputs(self):
        """Set text data to the output bundle"""
        outputs = []
        # indices = [np.arange(len(x)) for x in self.get_data()]
        # indices = Indices(instances=indices, roles=self.roles)


        text = Text(self.get_data(), self.vocabulary)
        dp = DataPack(text, usage=self.indices)
        outputs.append(dp)

        if self.is_labelled():
            labels_data = Numeric(self.labels)
            labels_usage = Labels(self.labelset, self.multilabel)
            dp = DataPack(labels_data, labels_usage)
            dp.add_usage(self.indices)
            outputs.append(dp)
        if self.targets:
            # tar = self.train_targets + self.test_targets
            # dp = DataPack.make(tar, GroundTruth)
            dp = DataPack.make(self.targets, GroundTruth)
            dp.add_usage(self.indices)
            outputs.append(dp)

        self.data_pool.add_data_packs(outputs, self.name)

    def get_component_inputs(self):
        # not needed for datasets
        pass

    def configure_name(self):
        self.name = Dataset.generate_name(self.config)
        if self.config.has_limit():
            self.sampler = Sampler(self.config)
            self.name = self.sampler.get_limited_name(self.name)
        else:
            self.name = Dataset.generate_name(self.config)
        Component.configure_name(self, self.name)


    # endregion

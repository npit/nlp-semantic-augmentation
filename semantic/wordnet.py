from os import listdir

import nltk

from semantic.semantic_resource import SemanticResource
from utils import info, nltk_download
from nltk.corpus import wordnet as wn
from collections import defaultdict


class Wordnet(SemanticResource):
    name = "wordnet"
    # pos_tag_mapping = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADJ, "RB": wn.ADV}
    # local name to synset cache for spreading activation speedups
    name_to_synset_cache = {}

    def initialize_lookup(self):
        if self.initialized:
            return
        Wordnet.setup_nltk_resources(self.config)
        self.initialized = True

    @staticmethod
    def get_wordnet_pos(config, treebank_tag):
        # Wordnet.setup_nltk_resources(config)
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return ''

    def get_clear_concept_word(self, concept):
        name = concept._name if type(concept) != str else concept
        if "." not in name:
            return name
        return name[: name.index(".")]

    def get_all_available_concepts(self):
        info("Getting all available {} concepts".format(self.name))
        return list(wn.all_synsets())

    def __init__(self, config):
        self.config = config
        self.initialized = False
        SemanticResource.__init__(self)

    def setup_nltk_resources(config):
        try:
            info("Setting up WordNet...")
            wn.VERB
        except:
            nltk_download(config, "wordnet")

    def fetch_raw(self, dummy_input):
        if self.base_name not in listdir(nltk.data.find("corpora")):
            nltk_download(self.config, "wordnet")
        return None

    def handle_raw_serialized(self, raw_serialized):
        pass

    def handle_raw(self, raw_data):
        pass

    def get_model(self):
        # save wordnet synset names to avoid depend
        return [s.name() for s in self.vocabulary]

    def load_model(self):
        """Default model loading function, via pickled object deserializaLoad the model"""
        if super().load_model():
            # map vocab to synset objects
            info("Mapping semantic vocabulary strings to wordnet synset objects")
            self.vocabulary = [wn.synset(t) for t in self.vocabulary]
        return self.model_loaded


    def analyze(self, inputs):
        """Analyzer function"""
        return self.tokenize(inputs)

    def tokenize(self, words):
        """Tokenizer function"""
        synsets = []
        for word in words:
            synsets.extend(self.get_word_synsets(word))
        return synsets

    def get_word_synsets(self, word):
        """Fetch synsets from an input word"""
        synsets = wn.synsets(word)
        if not synsets:
            return {}
        synsets = self.disambiguate(synsets, word)
        return synsets

    def spread_activation(self, synset_name):
        """Retrieve wordnet hypernyms from a given synset"""
        if synset_name in self.name_to_synset_cache:
            synset = self.name_to_synset_cache[synset_name]
        else:
            synset = wn.synset(synset_name)
        return [h._name for h in synset.hypernyms()]

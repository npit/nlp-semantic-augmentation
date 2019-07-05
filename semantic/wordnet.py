from os import listdir

import nltk

from semantic.semantic_resource import SemanticResource
from utils import info, nltk_download
from nltk.corpus import wordnet as wn


class Wordnet(SemanticResource):
    name = "wordnet"
    # pos_tag_mapping = {"VB": wn.VERB, "NN": wn.NOUN, "JJ": wn.ADJ, "RB": wn.ADV}
    # local name to synset cache for spreading activation speedups
    name_to_synset_cache = {}

    @staticmethod
    def get_wordnet_pos(config, treebank_tag):
        Wordnet.setup_nltk_resources(config)
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
        Wordnet.setup_nltk_resources(config)
        SemanticResource.__init__(self)

    def setup_nltk_resources(config):
        try:
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

    def lookup(self, word_information):
        word, _ = word_information
        synsets = wn.synsets(word)
        if not synsets:
            return {}
        synsets = self.disambiguate(synsets, word_information)
        activations = {synset._name: 1 for synset in synsets}
        return activations

    def spread_activation(self, synset_name):
        if synset_name in self.name_to_synset_cache:
            synset = self.name_to_synset_cache[synset_name]
        else:
            synset = wn.synset(synset_name)
        return [h._name for h in synset.hypernyms()]

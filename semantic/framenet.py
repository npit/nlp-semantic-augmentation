from os import listdir

import nltk

import defs
from semantic.semantic_resource import SemanticResource
from nltk.corpus import framenet as fn
from utils import nltk_download


class Framenet(SemanticResource):
    name = "framenet"
    relations_to_spread = ["Inheritance"]

    def __init__(self, config):
        self.config = config
        SemanticResource.__init__(self)
        # map nltk pos maps into meaningful framenet ones
        self.pos_tag_mapping = {"VB": "V", "NN": "N", "JJ": "A", "RB": "ADV"}


    def analyze(self, inputs):
        """Analyzer function"""
        concepts = []
        for word in inputs[:5]:
            concepts.extend(self.lookup(word))
        return concepts

    def initialize_lookup(self):
        if self.initialized:
            return
        try:
            fn.frames_by_lemma("dog")
        except LookupError:
            nltk_download(self.config, "framenet_v17")
        self.initialized = True

    def lookup(self, word):
        frames = fn.frames_by_lemma(word)
        return [f['name'] for f in frames]

    def lookup_(self, candidate):
        # http://www.nltk.org/howto/framenet.html
        word = candidate
        # in framenet, pos-disambiguation is done via the lookup
        if self.disambiguation == defs.disam.pos:
            frames = self.lookup_with_POS(candidate)
        else:
            frames = fn.frames_by_lemma(word)
            if not frames:
                return 
            frames = self.disambiguate(frames, candidate)
        if not frames:
            return None
        activations = {x.name: 1 for x in frames}
        if self.do_spread_activation:
            parent_activations = self.spread_activation(frames, self.spread_steps, 1)
            activations = {**activations, **parent_activations}
        return activations

    def lookup_with_POS(self, candidate):
        word, word_pos = candidate
        if word_pos in self.pos_tag_mapping:
            word += "." + self.pos_tag_mapping[word_pos]
        frames = fn.frames_by_lemma(word)
        if not frames:
            return None
        return self.disambiguate(frames, candidate, override=defs.disam.first)

    def get_related_frames(self, frame):
        # get just parents
        return [fr.Parent for fr in frame.frameRelations if fr.type.name == "Inheritance" and fr.Child == frame]

    def spread_activation(self, frames, steps_to_go, current_decay):
        if steps_to_go == 0:
            return
        activations = {}
        current_decay *= self.spread_decay_factor
        for frame in frames:
            related_frames = self.get_related_frames(frame)
            for rel in related_frames:
                activations[rel.name] = current_decay
                parents = self.spread_activation([rel], steps_to_go - 1, current_decay)
                if parents:
                    activations = {**activations, **parents}
        return activations


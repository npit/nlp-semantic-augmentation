# import gensim
from nltk.corpus import wordnet as wn

class Wordnet:
    name="wordnet"
    word_synset_cache = {}
    synset_freqs = []

    # function to map words to wordnet concepts
    def map_text(self, datasets_words):

        self.synset_freqs = [[] for _ in datasets_words]
        for d, dset in enumerate(datasets_words):
            for wl, word_list in enumerate(dset):
                # import pdb; pdb.set_trace()
                freqs = {}
                for w, word in enumerate(word_list):
                    print("Semantic processing for word {}/{} ({}) of doc {}/{} and dset {}/{}".format(w+1, len(word_list), word, wl+1, len(dset), d+1, len(datasets_words)))
                    synset, freqs = self.get_synset(word, freqs)
                    if not synset: continue
                    self.process_synset(synset)
                self.synset_freqs[d].append(freqs)


    # function to get a synset from a word, using the wordnet api
    # and a local word cache. Updates synset frequencies as well.
    def get_synset(self, word, freqs):
        if word in self.word_synset_cache:
            # print("Cache hit:", self.word_synset_cache)
            # print("freqs:", freqs)
            synset = self.word_synset_cache[word]
            if synset not in freqs:
                freqs[synset] = 0
        else:
            # print("Cache miss:", self.word_synset_cache)
            # print("freqs:", freqs)
            synsets = wn.synsets(word)
            if not synsets:
                return None, freqs
            synset = synsets[0]
            freqs[synset] = 0
            self.word_synset_cache[word] = synset

        freqs[synset] += 1
        return synset, freqs

    # function that applies the required processing
    # once a synset has been found in the input text
    def process_synset(self, synset):
        # spreading activation
        pass


    # function that applies post-processing for a collected synset graph 
    def postprocess_synset(self, synset):
        # frequency / tf-idf filtering - do that with synset freqs
        # ingoing / outgoing graph ratio
        pass

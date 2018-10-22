import pickle
from nltk.corpus import wordnet as wn, stopwords
from keras.preprocessing.text import text_to_word_sequence

"""
Script to extract words related to wordnet synsets through accompanying examples or definition
"""

# preprocess string to word list
def preprocess_text(text):
        stopw = set(stopwords.words('english'))
        filter = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\t1234567890'
        processed = text_to_word_sequence(text, filters=filter, lower=True, split=' ')
        processed = [p for p in processed if p not in stopw]
        return processed

# print example-wise synset frequencies
def example_freqs(thedict):
        print("Frequency of number of examples per synset:")
        freqs = [len(thedict[s]) for s in thedict]
        hist = {f:freqs.count(f) for f in freqs}
        total = sum(hist.values())
        for val, count in sorted(hist.items(), key = lambda p : p[1], reverse=True):
                print(val, count, "{:.3f} %".format(count/total*100))
# print word-wise synset frequencies
def word_freqs(thedict):
        print("Frequency of number of words per synset:")
        freqs = [sum([len(sent) for sent in thedict[s]]) for s in thedict]
        hist = {f:freqs.count(f) for f in freqs}
        total = sum(hist.values())
        for val, count in sorted(hist.items(), key = lambda p : p[1], reverse=True):
                print(val, count, "{:.3f} %".format(count/total*100))



examples_per_synset = {}
def_per_synset = {}
examples_def_per_synset = {}

# get all to know how much it is
all_ssets = list(wn.all_synsets())
# process each
for i, ss in enumerate(all_ssets):
    name = ss.name()
    examples = list(set([word for t in ss.examples() for word in preprocess_text(t)]))
    definition = list(set(preprocess_text(ss.definition())))
    print("Synset {}/{}".format(i+1, len(all_ssets)))

    if examples:
        examples_per_synset[name] = ss.examples()
        if ss.definition():
            examples_def_per_synset[name] = list(set(examples + definition))
    if definition:
        def_per_synset[name] = definition

# print results and save word lists
print("Examples exist for {}/{} synsets.".format(len(examples_per_synset), len(all_ssets)))
example_freqs(examples_per_synset)
word_freqs(examples_per_synset)
with open("wordnet_synset_examples.pickle", "wb") as f:
    pickle.dump(examples_per_synset, f)
print("Examples and definitions exist for {}/{} synsets.".format(len(examples_def_per_synset), len(all_ssets)))
word_freqs(examples_def_per_synset)
with open("wordnet_synset_examples_definitions.pickle", "wb") as f:
    pickle.dump(examples_def_per_synset, f)
print("Definitions exist for {}/{} synsets.".format(len(def_per_synset), len(all_ssets)))
word_freqs(def_per_synset)
with open("wordnet_synset_definitions.pickle", "wb") as f:
    pickle.dump(def_per_synset, f)

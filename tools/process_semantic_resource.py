import pickle
import tqdm
import yaml
import argparse
from nltk.corpus import wordnet as wn
from semantic.semantic_resource import SemanticResource
import dataset
import settings
from utils import to_namedtuple, info, setup_simple_logging

"""
Script to preprocess a semantic information resource for text classification.

"""


# print example-wise synset frequencies
def example_freqs(thedict):
    print("Frequency of number of examples per synset:")
    freqs = [len(thedict[s]) for s in thedict]
    hist = {f: freqs.count(f) for f in freqs}
    total = sum(hist.values())
    residuals, num_residuals = 0, 0
    # with tqdm.tqdm(total=len(hist), ascii=True, desc="Number of examples / synset") as pbar:
    for val, count in sorted(hist.items(), key=lambda p: p[1], reverse=True):
        pcnt = count / total * 100
        if pcnt > 2:
            print("value: {}, count: {}, pcnt: {:.3f} %".format(val, count, pcnt))
        else:
            num_residuals += 1
            residuals += pcnt

            # pbar.update()
    print("Total {} residuals, each <= 2 %: {}".format(num_residuals, residuals))


# print word-wise synset frequencies
def word_freqs(thedict):
    print("Frequency of number of words per synset:")
    freqs = [sum([len(sent) for sent in thedict[s]]) for s in thedict]
    hist = {f: freqs.count(f) for f in freqs}
    total = sum(hist.values())
    residuals, num_residuals = 0, 0
    # with tqdm.tqdm(total=len(hist), ascii=True, desc="Number of examples / synset") as pbar:
    for val, count in sorted(hist.items(), key=lambda p: p[1], reverse=True):
        pcnt = count / total * 100
        if pcnt > 2:
            print("value: {}, count: {}, pcnt: {:.3f} %".format(val, count, pcnt))
        else:
            num_residuals += 1
            residuals += pcnt

            # pbar.update()
    print("Total {} residuals, each <= 2 %: {}".format(num_residuals, residuals))


def mine_wordnet_examples_definitions():
    """ Extract words related to wordnet synsets through examples and/or definition of synsets
    """
    setup_simple_logging()
    examples_per_synset = {}
    def_per_synset = {}
    examples_def_per_synset = {}

    # instantiate dset object for text processing
    dset = dataset.Dataset(skip_init=True)
    dset.language = "english"
    dset.setup_nltk_resources()

    # get all to know how much it is

    info("Fetching all synsets.")
    all_ssets = list(wn.all_synsets())
    # process each
    with tqdm.tqdm(total=len(all_ssets), ascii=True, desc="Fetching example and definition words") as pbar:
        for i, ss in enumerate(all_ssets):
            name = ss.name()
            examples = list(set([word for t in ss.examples() for word in dset.process_single_text(t, dset.punctuation_remover, dset.digit_remover, dset.word_prepro, dset.stopwords)]))
            definition = list(set(dset.process_single_text(ss.definition(), dset.punctuation_remover, dset.word_prepro, dset.stopwords)))
            # print("Synset {}/{}".format(i + 1, len(all_ssets)))

            if examples:
                examples_per_synset[name] = ss.examples()
                if ss.definition():
                    examples_def_per_synset[name] = list(set(examples + definition))
            if definition:
                def_per_synset[name] = definition
            pbar.update()

    # print results and save word lists
    print("Examples exist for {}/{} synsets.".format(len(examples_per_synset), len(all_ssets)))
    print("Definitions exist for {}/{} synsets.".format(len(def_per_synset), len(all_ssets)))
    print("Examples or definitions exist for {}/{} synsets.".format(len(examples_def_per_synset), len(all_ssets)))
    example_freqs(examples_per_synset)
    # count word frequencies
    word_freqs(examples_per_synset)
    word_freqs(def_per_synset)
    word_freqs(examples_def_per_synset)

    # write results
    outfile = "wordnet_synset_examples.pickle"
    print("Writing to {}".format(outfile))
    with open(outfile, "wb") as f:
        pickle.dump(examples_per_synset, f)

    print("Writing to {}".format(outfile))
    outfile = "wordnet_synset_examples_definitions.pickle"
    with open(outfile, "wb") as f:
        pickle.dump(examples_def_per_synset, f)

    outfile = "wordnet_synset_definitions.pickle"
    print("Writing to {}".format(outfile))
    with open(outfile, "wb") as f:
        pickle.dump(def_per_synset, f)


def produce_semantic_neighbourhood(config_file):
    """Function that produces semantic word neighbourhoods from a semantic resource.
    """
    try:
        config = settings.Config(config_file)
        config.misc.independent_component = True
        config.misc.deserialization_allowed = False
        config.semantic.spreading_activation = config.semantic.spreading_activation[0], 0.5
        semres = SemanticResource.create(config)
    except:
        print("Problematic configuration in {}".format(config_file))

    all_concepts = semres.get_all_available_concepts()
    info("Got a total of {} concepts".format(semres.name))
    neighbours = {}
    info("Getting semantic neighbours for up to {} steps ".format(semres.spread_steps))
    with tqdm.tqdm(total=len(all_concepts), ascii=True) as pbar:
        for concept in all_concepts:
            sr = semres.spread_activation([concept], semres.spread_steps, 1)
            if sr:
                clear_name = semres.get_clear_concept_word(concept)
                if clear_name not in neighbours:
                    neighbours[clear_name] = {}
                items = list(sr.items())
                step_index = 0
                while items:
                    max_w = max(items, key=lambda x: x[1])[1]
                    idxs = [i for i in range(len(items)) if items[i][1] == max_w]
                    if step_index not in neighbours[clear_name]:
                        neighbours[clear_name][step_index] = []
                    neighbours[clear_name][step_index].extend([semres.get_clear_concept_word(items[i][0]) for i in idxs])
                    items = [items[j] for j in range(len(items)) if j not in idxs]
                    step_index += 1
            pbar.update()

    # write marginal
    for step in range(semres.spread_steps):
        outfile = "{}.neighbours.step_{}.txt".format(semres.base_name, step + 1)
        info("Writing to {}".format(outfile))
        with open(outfile, "w") as f:
            for concept in neighbours:
                if step in neighbours[concept]:
                    neighs = list(set(neighbours[concept][step]))
                    f.write("{} {}\n".format(concept, " ".join(neighs)))
    # write total
    outfile = "{}.neighbours.total.txt".format(semres.base_name, step + 1)
    info("Writing to {}".format(outfile))
    with open(outfile, "w") as f:
        for c, concept in enumerate(neighbours):
            neighs = []
            for step in range(semres.spread_steps):
                if step in neighbours[concept]:
                    neighs.extend(neighbours[concept][step])
            neighs = list(set(neighs))
            f.write("{} {}\n".format(concept, " ".join(neighs)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config_file', required=True)
    return parser.parse_args()


def main():

    args = parse_arguments()
    produce_semantic_neighbourhood(args.config_file)


if __name__ == '__main__':
    main()

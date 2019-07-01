import numpy as np
import pandas
import os
from os.path import join, exists
import nltk
import yaml
from utils import write_pickled, warning, nltk_download
import argparse

from representation import word_embedding, bag_representation
import semantic
import transform
from defs import alias

import large_scale


""" Run tests
"""


def setup_test_resources(args):
    """Creates the necessary data and configuration for running tests.
    """

    config_file = join(args.test_directory, "test.config.yml")
    os.makedirs(args.test_directory, exist_ok=True)
    csv_separator = ","

    if args.cont:
        print("Resuming testing in directory {}".format(args.test_directory))
        return config_file
    nltk.data.path = [args.nltk]

    print("Creating test data and configuration")
    num_words = 100
    embedding_dim = 30
    serialization_dir = "serialization_test"
    raw_data_dir = "raw_data_test"

    if not exists(join(raw_data_dir, "representation")):
        os.makedirs(join(raw_data_dir, "representation"))
    if exists(serialization_dir):
        warning("Note: serialization directory already exists:{}".format(serialization_dir))

    print("Creating dummy embedding mapping.")
    try:
        words = nltk.corpus.words.words()
    except LookupError:
        nltk_download(None, "words")
        words = nltk.corpus.words.words()

    # pick random words for the embedding mapping
    words = np.random.choice(words, num_words)
    # make random embedding
    embedding_map = np.random.rand(len(words), embedding_dim)
    pandas.DataFrame(embedding_map, index=words).to_csv(
        join(raw_data_dir, "representation", "word_embedding.csv"), header=None, sep=csv_separator)

    # write large-scale yaml config
    with open("example.config.yml") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # delete param fields
    # insert available configurations
    conf["params"] = {}

    conf["params"]["representation"] = {"name": [word_embedding.WordEmbedding.name, bag_representation.BagRepresentation.name, bag_representation.TFIDFRepresentation.name],
                                        "aggregation": ["avg", "pad", "none"], "sequence_length": [1, 10]}
    del conf["representation"]["name"]
    del conf["representation"]["aggregation"]
    del conf["representation"]["sequence_length"]

    # drop lida due to colinear results -- should be testing manually
    conf["params"]["transform"] = {"name": [t for t in transform.Transform.get_available() if t != transform.LiDA.base_name] + [alias.none]}
    del conf["transform"]["name"]

    # conf["params"]["semantic"] = semantic.SemanticResource.get_available()
    conf["params"]["semantic"] = {"name": [semantic.Wordnet.name, alias.none]}
    del conf["semantic"]["name"]

    conf["params"]["learning"] = {"name": ["mlp"], "hidden_dim": [64], "layers": [2]}
    for x in ["name", "hidden_dim", "layers"]:
        del conf["learning"][x]


    # set static parameters
    conf["dataset"] = {"name": "20newsgroups", "data_limit": [190, 90], "class_limit": 6}
    conf["experiments"] = {"run_folder": args.test_directory}
    conf["folders"]["serialization"] = serialization_dir
    conf["folders"]["raw_data"] = raw_data_dir
    conf["representation"]["dimension"] = embedding_dim
    conf["representation"]["limit"] = ["top", 30]
    conf["transform"]["dimension"] = 4
    conf["semantic"]["limit"] = ["top", 20]
    conf["semantic"]["disambiguation"] = "first"

    conf["train"]["folds"] = 3
    conf["train"]["validation_portion"] = None

    conf["print"]["run_types"] = ["run", "majority"]
    conf["learning"]["no_load"] = True

    # write incompatible combinations
    write_bad_combos(config_file + ".bad_combos")

    with open(config_file, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)
    print("Generated test yaml config at {}".format(config_file))
    return config_file

# write down inconsistent parameter configs to skip during testing
def write_bad_combos(outfile):
    combos = []
    # bags with aggregation
    combos.append(
        [
        [["representation", "name"], "bag"],
        [["representation", "aggregation"], ["avg", "pad"]]
            ])
    combos.append(
        [
        [["representation", "name"], "tfidf"],
        [["representation", "aggregation"], ["avg", "pad"]]
            ])
    # lstms without pad
    combos.append(
        [
        [["learning", "name"], "lstm"],
        [["representation", "aggregation"], ["avg", "none"]]
            ])
    # pad with no lstms
    combos.append(
        [
        [["learning", "name"], ["mlp", "kmeans", "naive_bayes"] ],
        [["representation", "aggregation"], "pad"]
            ])
    # WEs with no aggregation
    combos.append(
        [
        [["representation", "name"], ["word_embedding"] ],
        [["representation", "aggregation"], "none"]
            ])
    # aggregation that results in a single vector, with non-unit sequence_length
    combos.append(
        [
        [["representation", "aggregation"], ["avg"] ],
        [["representation", "sequence_length"], 10]
            ])

    with open(outfile, "w") as f:
        yaml.dump(combos, f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing script for the semantic augmentation tool.')
    parser.add_argument('-c', '--continue', action="store_true", dest="cont")
    parser.add_argument('-g', '--generate-only', action="store_true")
    parser.add_argument('-d', '--directory', dest="test_directory", default="test_runs")
    parser.add_argument('--nltk', default="raw_data_test", help="nltk data directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    conf_file = setup_test_resources(args)
    if args.generate_only:
        print("Generated, exiting.")
        exit(1)
    large_scale.main(conf_file, is_testing_run=True)
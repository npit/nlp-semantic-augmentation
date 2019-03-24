import numpy as np
import pandas
import os
from os.path import join, exists
import nltk
import yaml
from utils import write_pickled, warning
import argparse

import representation
import semantic
import transform
from defs import alias

import large_scale


""" Run tests
"""


def setup_test_resources(args):
    """Creates the necessary data and configuration for running tests.
    """

    config_file = "test.config.yml"

    if args.cont:
        print("Resuming testing in directory {}".format(args.test_directory))
        return join(args.test_directory, config_file)

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
        nltk.download("words")
        words = nltk.corpus.words.words()

    # pick random words for the embedding mapping
    words = np.random.choice(words, num_words)
    # make random embedding
    embedding_map = np.random.rand(len(words), embedding_dim)
    pandas.DataFrame(embedding_map, index=words).to_csv(
        join(raw_data_dir, "representation", "vector_embedding.csv"), header=None, sep=" ")

    # write large-scale yaml config
    with open("example.config.yml") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # insert available configurations
    conf["params"] = {}
    conf["params"]["representation"] = {"name": [representation.VectorEmbedding.name, representation.BagRepresentation.name, representation.TFIDFRepresentation.name]}
    # drop lida due to colinear results -- should be testing manually
    conf["params"]["transform"] = {"name": [t for t in transform.Transform.get_available() if t != transform.LiDA.base_name] + [alias.none]}
    # conf["params"]["semantic"] = semantic.SemanticResource.get_available()
    conf["params"]["semantic"] = {"name": [semantic.Wordnet.name, alias.none]}
    conf["params"]["learner"] = {"name": ["mlp"], "hidden_dim": [64], "layers": [2], "no_load": [True]}

    # set static parameters
    conf["dataset"] = {"name": "20newsgroups", "data_limit": [300, 150], "class_limit": 6}
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

    with open(config_file, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)
    return config_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing script for the semantic augmentation tool.')
    parser.add_argument('-c', '--continue', action="store_true", dest="cont")
    parser.add_argument('-d', '--directory', dest="test_directory", default="test_runs")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    conf_file = setup_test_resources(args)
    large_scale.main(conf_file)

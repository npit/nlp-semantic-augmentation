#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import keras
#  wordnet api in nltk
# from nltk.corpora import wordnet as wn
# import numpy as np

from fetcher import Fetcher
from helpers import Config
import argparse


print("Imports done.")


def main(config_file):
    print("Running main.")
    # initialize configuration
    config = Config()
    config.initialize(config_file)
    fetcher = Fetcher()

    # datasets loading & preprocessing
    dataset = fetcher.fetch_dataset(config.get_dataset())
    dataset.make(config)
    dataset.preprocess()

    # embedding
    embedding = fetcher.fetch_embedding(config.get_embedding())
    embedding.map_text(dataset)

    # semantic enrichment
    semantic = fetcher.fetch_semantic(config.get_semantic_resource())
    semantic.map_text(embedding.get_words(), dataset.get_name())

    # learning
    learner = fetcher.fetch_learner(config.get_learner())
    learner.make(embedding.get_data(), dataset.get_targets(), config)

    learner.do_train()
    learner.do_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file
    main(config_file)

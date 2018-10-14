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
import logging


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
    embedding.make(config)
    embedding.map_text(dataset, config)
    embedding.prepare(config)

    # semantic enrichment
    semantic = fetcher.fetch_semantic(config.get_semantic_resource())
    semantic.make(config)
    semantic.map_text(embedding.get_words(), dataset.get_name())
    embedding.enrich(semantic.get_data(config), config)

    # learning
    # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
    learner = fetcher.fetch_learner(config.get_learner())
    learner.make(embedding.get_data(), dataset.get_targets(), dataset.get_num_labels(), config)

    learner.do_traintest(config)
    logging.getLogger().info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file
    main(config_file)

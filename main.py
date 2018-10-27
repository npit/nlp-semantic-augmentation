#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# import keras
#  wordnet api in nltk
# from nltk.corpora import wordnet as wn
# import numpy as np

from fetcher import Fetcher
from dataset import Dataset
from embedding import Embedding
from dataset import Dataset
from helpers import Config
import argparse
import logging
from utils import info
from semantic import SemanticResource


print("Imports done.")

print("googlesemantic, spreading act")

def main(config_file):
    print("Running main.")
    # initialize configuration
    config = Config()
    config.initialize(config_file)
    fetcher = Fetcher()

    # datasets loading & preprocessing
    info("===== DATASET =====")
    dataset = Dataset.create(config)

    # embedding
    info("===== EMBEDDING =====")
    embedding = Embedding.create(config)
    embedding.map_text(dataset)
    embedding.prepare()

    # semantic enrichment
    semantic_data = None
    if config.has_enrichment():
        info("===== SEMANTIC =====")
        semantic = SemanticResource.create(config)
        semantic.map_text(embedding)
        semantic_data = semantic.get_data(config)
    embedding.finalize(semantic_data)

    # learning
    info("===== LEARNING =====")
    # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
    learner = DNN.create(config)
    learner.make(embedding, dataset.get_targets(), dataset.get_num_labels(), config)

    learner.do_traintest(config)
    info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file
    main(config_file)

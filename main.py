#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dataset import Dataset
from embedding import Embedding
from semantic import SemanticResource, GoogleKnowledgeGraph
from learner import DNN
from helpers import Config
import argparse
from utils import info
print("Imports done.")

def main(config_file):

    print("Running main.")
    # initialize configuration
    config = Config()
    config.initialize(config_file)

    #gkg = GoogleKnowledgeGraph(config)
    #gkg.lookup("dog")

    # datasets loading & preprocessing
    info("===== DATASET =====")
    dataset = Dataset.create(config)

    # embedding
    info("===== EMBEDDING =====")
    embedding = Embedding.create(config)
    embedding.map_text(dataset)
    embedding.prepare()

    semantic = None
    # semantic enrichment
    if config.has_semantic():
        info("===== SEMANTIC =====")
        semantic = SemanticResource.create(config)
        semantic.map_text(embedding, dataset)
        semantic.generate_vectors()
    embedding.finalize(semantic)

    # learning
    info("===== LEARNING =====")
    # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
    learner = DNN.create(config)
    learner.make(embedding, dataset.get_targets(), dataset.get_num_labels())
    learner.do_traintest()

    info("Logfile is at: {}".format(config.logfile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file
    main(config_file)

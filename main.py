#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
import warnings
from dataset import Dataset
from representation import Representation
from semantic import SemanticResource
from transform import Transform
from learner import DNN
from helpers import Config
import argparse
from utils import info
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Imports done.")


def main(config_file):

    print("Running main.")
    # initialize configuration
    config = Config()
    config.initialize(config_file)

    # datasets loading & preprocessing
    info("===== DATASET =====")
    dataset = Dataset.create(config)

    # embedding
    info("===== EMBEDDING =====")
    representation = Representation.create(config)
    representation.map_text(dataset)
    representation.compute_dense()

    transform = None
    if config.has_transform():
        info("===== TRANSFORM =====")
        transform = Transform.create(config)
        transform.compute(representation.get_vectors())
        representation.set_transformed(transform)

    representation.aggregate_instance_vectors()
    semantic = None
    # semantic enrichment
    if config.has_semantic():
        info("===== SEMANTIC =====")
        semantic = SemanticResource.create(config)
        semantic.map_text(representation, dataset)
        semantic.generate_vectors()

    representation.finalize(semantic, transform)

    # learning
    info("===== LEARNING =====")
    # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
    learner = DNN.create(config)
    learner.make(representation, dataset.get_targets(), dataset.get_num_labels())
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

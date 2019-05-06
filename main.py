#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
import warnings
from dataset import Dataset
from representation import Representation
from semantic import SemanticResource
from transform import Transform
from settings import Config
import argparse
from utils import info, warning, num_warnings, tictoc
from instantiator import instantiate_learner
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Imports done.")


def main(config_file):

    print("Running main.")
    # initialize configuration
    config = Config(config_file)

    # time the entire run
    with tictoc("Total run"):
        # datasets loading & preprocessing
        info("===== DATASET =====")
        dataset = Dataset.create(config)
        dataset.preprocess()

        # check for existing & precomputed transformed representations
        info("===== REPRESENTATION =====")
        # setup the representation
        representation = Representation.create(config)

        transform = None
        if config.has_transform():
            transform = Transform.create(representation)
            if not transform.loaded():
                # load / acquire necessary data for computation
                representation.acquire_data()
            else:
                # transfer loaded data from the transformed bundle
                representation.set_transform(transform)

        # representation computation
        if not config.has_transform() or not transform.loaded():
            representation.map_text(dataset)
            representation.compute_dense()

        # transform computation
        if config.has_transform():
            info("===== TRANSFORM =====")
            transform.compute(representation, dataset)

        # aggregation
        representation.aggregate_instance_vectors()

        # semantic enrichment
        semantic = None
        if config.has_semantic():
            info("===== SEMANTIC =====")
            semantic = SemanticResource.create(config)
            semantic.map_text(representation, dataset)
            semantic.generate_vectors()
            representation.set_semantic(semantic)

        # learning
        info("===== LEARNING =====")
        # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
        learner = instantiate_learner(config)
        learner.make(representation, dataset)
        learner.do_traintest()

        if num_warnings > 0:
            warning("{} warnings occured.".format(num_warnings - 1))
        info("Logfile is at: {}".format(config.logfile))
    tictoc.log(config.logfile + ".timings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file

    main(config_file)

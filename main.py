#! /home/nik/work/iit/submissions/NLE-special/venv/bin/python3.6
from dataset import instantiator as dset_instantiator
from representation import instantiator as rep_instantiator
from semantic import instantiator as sem_instantiator
from transform.transform import Transform
from learning import instantiator as lrn_instantiator

from settings import Config
import argparse
from utils import info, warning, num_warnings, tictoc, error


def main(config_file):

    # initialize configuration
    conf = Config(config_file)
    pipeline = conf.get_pipeline()

    pipeline.configure_names()
    pipeline.run()
    # error("Add a required datatypes per component")

    #
    # # time the entire run
    # with tictoc("Total run"):
    #     # datasets loading & preprocessing
    #     info("===== DATASET =====")
    #     dataset = dset_instantiator.create(config)
    #     dataset.preprocess()
    #
    #     # check for existing & precomputed transformed representations
    #     info("===== REPRESENTATION =====")
    #     # setup the representation
    #     representation = rep_instantiator.create(config)
    #
    #     transform = None
    #     if config.has_transform():
    #         transform = Transform.create(representation)
    #         if not transform.loaded():
    #             # load / acquire necessary data for computation
    #             representation.acquire_data()
    #         else:
    #             # transfer loaded data from the transformed bundle
    #             representation.set_transform(transform)
    #
    #     # representation computation
    #     if not config.has_transform() or not transform.loaded():
    #         representation.map_text(dataset)
    #         representation.compute_dense()
    #
    #     # transform computation
    #     if config.has_transform():
    #         info("===== TRANSFORM =====")
    #         transform.compute(representation, dataset)
    #
    #     # aggregation
    #     representation.aggregate_instance_vectors()
    #
    #     # semantic enrichment
    #     semantic = None
    #     if config.has_semantic():
    #         info("===== SEMANTIC =====")
    #         semantic = sem_instantiator.create(config)
    #         semantic.map_text(representation, dataset)
    #         semantic.generate_vectors()
    #         representation.set_semantic(semantic)
    #
    #     # learning
    #     info("===== LEARNING =====")
    #     # https: // blog.keras.io / using - pre - trained - word - embeddings - in -a - keras - model.html
    #     learner = lrn_instantiator.create(config)
    #     learner.make(representation, dataset)
    #     learner.do_traintest()
    #
    #     if num_warnings > 0:
    #         warning("{} warnings occured.".format(num_warnings - 1))
    #     info("Logfile is at: {}".format(config.logfile))
    # tictoc.log(config.logfile + ".timings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Configuration .yml file for the run.")
    args = parser.parse_args()
    if args.config_file is None:
        config_file = "config.yml"
    else:
        config_file = args.config_file

    main(config_file)

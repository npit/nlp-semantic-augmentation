"""The entrypoint module"""

import argparse

from dataset import instantiator as dset_instantiator
from learning import instantiator as lrn_instantiator
from representation import instantiator as rep_instantiator
from semantic import instantiator as sem_instantiator
from settings import Config
from transform.transform import Transform
from utils import error, info, num_warnings, tictoc, warning


def main(config_file):
    """The main function

    Arguments:
        config_file {str} -- Path for the run's configuration file
    """
    # # time the entire run
    with tictoc("Total run"):
        # initialize configuration
        conf = Config(config_file)
        pipeline = conf.get_pipeline()

        pipeline.configure_names()
        pipeline.run()

        if num_warnings > 0:
            warning("{} warnings occured.".format(num_warnings - 1))
        info("Logfile is at: {}".format(conf.logfile))
    tictoc.log(conf.logfile + ".timings")


if __name__ == "__main__":
    """Top-level entrypoint code block"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Configuration .yml file for the run.", nargs="?", default="chain.config.yml")
    args = parser.parse_args()
    main(args.config_file)

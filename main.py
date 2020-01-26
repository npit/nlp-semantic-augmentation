"""The entrypoint module"""
import argparse

from settings import Config
from utils import info, num_warnings, tictoc, warning


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
    # Top-level entrypoint code block
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Configuration .yml file for the run.", nargs="?")
    args = parser.parse_args()
    main(args.config_file)

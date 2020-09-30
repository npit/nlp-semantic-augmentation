"""The entrypoint module"""
import argparse

from config.config_reader import ConfigReader
from utils import info, num_warnings, tictoc, warning


def main(config_file, ignore_undefined=False, load_models_first=False):
    """The main function

    Arguments:
        config_file {str} -- Path for the run's configuration file
    """
    # # time the entire run
    with tictoc("Total run"):
        # initialize configuration
        global_config, pipeline, triggers = ConfigReader.read_configuration(config_file, ignore_undefined)

        pipeline.configure_names()

        # 
        should_load_models = load_models_first or any(trig.requires_model_loading() for trig in triggers)
        if should_load_models:
            pipeline.load_models()

        for trig in sorted(triggers, key=lambda x: x.is_blocking):
            trig.link_pipeline(pipeline)
            trig.setup()

        for trig in triggers:
            trig.arm()

        if num_warnings > 0:
            warning("{} warnings occured.".format(num_warnings - 1))
        info("Logfile is at: {}".format(global_config.logfile))
    tictoc.log(global_config.logfile + ".timings")


if __name__ == "__main__":
    # Top-level entrypoint code block
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Configuration .yml file for the run.", nargs="?")
    parser.add_argument("--load_models", help="Load models prior to first pipeline execution.", action="store_true", default=False)
    parser.add_argument("--ignore-undefined-keys", help="Ignore undefined keys in the configuration file.", action="store_true", dest="ignore_undefined")
    args = parser.parse_args()
    main(args.config_file, args.ignore_undefined, load_models_first=args.load_models)

"""Module for defining global-level configuration settings and components
"""
import logging
from os.path import join

from config.config import Configuration
from config.global_components import global_component_classes

class GlobalConfig(Configuration):
    """Global configuration class
    """
    # global-level configuration bundles
    misc = None
    print = None
    folders = None
    # key for declaring chains
    chains_key = "chains"
    triggers_key = "triggers"


    def __init__(self):
        super().__init__(None)

    def finalize(self):
        """
        Finalize the configuration, filling in non-determined attributes with defaults
        """
        for cl in global_component_classes:
            if self.__getattribute__(cl.conf_key_name) is None:
                self.add_config_object(cl.conf_key_name, cl())

    # logging initialization
    def setup_logging(self):
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)7s | %(message)s')

        # console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        lvl = logging._nameToLevel[self.print.log_level.upper()]
        # logger = logging.getLogger(self.logger_name)
        logger = logging.getLogger()
        logger.setLevel(lvl)
        logger.addHandler(handler)

        # file handler
        self.logfile = join(self.folders.run, "log_{}.log".format(self.misc.run_id))
        fhandler = logging.FileHandler(self.logfile)
        fhandler.setLevel(lvl)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

        self.logger = logger
        # remove extraneous handlers
        handlers_to_remove = []
        for h in self.logger.handlers:
            if h not in [fhandler, handler]:
                handlers_to_remove.append(h)
        for h in handlers_to_remove:
            self.logger.removeHandler(h)

        return logger

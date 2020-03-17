import os
import shutil
from os.path import exists, join

import utils
from component.chain import Chain
from component.pipeline import Pipeline
from config.chain_components import chain_component_classes
from config.global_components import global_component_classes
from config.global_config import GlobalConfig
from utils import debug, error


class ConfigReader:
    @staticmethod
    def read_configuration(conf_file=None):
        """Configuration object constructor
        """
        # read the configuration file
        conf_dict = utils.read_ordered_yaml(conf_file)

        # read non-chain configuration
        global_config = ConfigReader.read_global_configuration(conf_dict)
        # copy configuration to the run folder
        run_config_path = join(global_config.folders.run, os.path.basename(conf_file))
        if not exists(run_config_path):
            shutil.copy(conf_file, run_config_path)

        # setup logging now to document subsequent operations
        global_config.setup_logging()
        # read chains
        pipeline = ConfigReader.read_pipeline(conf_dict, global_config)
        return global_config, pipeline

    @staticmethod
    def read_global_configuration(config_dict):
        """Read global-level configuration (accessible to all components and chains)
        Arguments:
            input_config {dict} -- The configuration
        """
        global_config = GlobalConfig()
        for global_conf_key, comp_conf in config_dict.items():
            if global_conf_key == GlobalConfig.chains_key:
                continue
            comp = [g for g in global_component_classes if g.conf_key_name == global_conf_key]
            if len(comp) != 1:
                error(f"Undefined global configuration component: {global_conf_key}", not comp)
                error(f"Multiple global configuration component matches: {global_conf_key}", len(comp) > 1)
            comp = comp[0]
            component_config = comp(comp_conf)
            global_config.add_config_object(global_conf_key, component_config)

        if global_config.misc.run_id is None:
            global_config.misc.run_id = utils.datetime_str()
        # make directories
        os.makedirs(global_config.folders.run, exist_ok=True)
        os.makedirs(global_config.folders.raw_data, exist_ok=True)
        os.makedirs(global_config.folders.serialization, exist_ok=True)
        return global_config

    @staticmethod
    def read_pipeline(chains_config, global_config):
        """Read all chains defined in the configuration
        Arguments:
            chains_config {dict} -- The configuration
        """
        # create the pipeline object to instantiate chain components on
        pipeline = Pipeline()
        chains_key = GlobalConfig.chains_key
        if chains_key not in chains_config:
            error("Configuration lacks chains information (key: {chains_key}")

        chains = chains_config[chains_key]
        for chain_name, chain_dict in chains.items():
            # read chain configuration dict
            component_names, component_configs = ConfigReader.read_chain_components(chain_dict, global_config)
            # build the chain object
            chain = Chain(chain_name, component_names, component_configs)
            # add to the pipeline
            pipeline.add_chain(chain)
        return pipeline

    def read_chain_components(chain_config, global_config):
        """Read configuration for a single chain
        """
        components, component_names = [], []
        valid_component_names = [c.conf_key_name for c in chain_component_classes]
        for component_name, component_dict in chain_config.items():
            if component_name not in valid_component_names:
                error(f"Undefined component name {component_name}. Available are {valid_component_names}")

            # valid component key encountered; create it
            component_class = chain_component_classes[valid_component_names.index(component_name)]
            debug(f"Read configuration for chain component {component_name}")
            component_config = component_class(component_dict)

            # merge the global configuration to the component configuration
            component_config.merge_other_config(global_config)
            components.append(component_config)
            component_names.append(component_name)
        return component_names, components

"""Dataset instantiation module"""
from dataset.manual import ManualDataset
from dataset.input_string import InputString
from dataset.reuters import Reuters
from dataset.twenty_newsgroups import TwentyNewsGroups


class Instantiator:
    """Class to instantiate a dataset object"""
    component_name = "dataset"

    def create(config):
        name = config.name
        if name == TwentyNewsGroups.name:
            return TwentyNewsGroups(config)
        elif name == Reuters.name:
            return Reuters(config)
        elif name == InputString.name:
            return InputString(config)
        else:
            # default to manually-defined dataset
            return ManualDataset(config)

from dataset.manual import ManualDataset
from dataset.manual_reader import ManualDatasetReader
from bundle.datausages import *
from bundle.datatypes import *

class InputString(ManualDataset):
    """Class to represent a string dataset"""
    name = "string"

    consumes = Text.name

    def __init__(self, config):
        self.config = config
        ManualDataset.__init__(self, config)

    def apply_dataset_reader(self, data):
        """Handle minimalist string input"""
        mdr = ManualDatasetReader()
        try:
            data = mdr.instances_to_json_dataset(data)
        except ValueError as v:
            error(str(v))
        except KeyError as v:
            error(str(v))
        mdr.read_dataset(raw_data=data)
        return mdr

    def load_model_from_disk(self):
        # for datasets, equivalent to loading the dataset
        return True

    def load_outputs_from_disk(self):
        """String input dataset is not deserializable"""
        self.set_serialization_params()
        return False

    def get_component_inputs(self):
        """String input dataset can get inputs"""
        raw_string = self.data_pool.request_data(Text, usage=Indices, client=self.name, reference_data=self.data_pool.data)
        strings = raw_string.data.instances
        self.handle_raw(strings)

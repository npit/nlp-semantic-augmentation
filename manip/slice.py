
# from manip.filter import Filter
from manip.manip import Manipulation
from utils import info, error, is_collection
from collections import OrderedDict
from bundle.datatypes import *
from bundle.datausages import *
from dataset.dataset import Dataset

class Slice(Manipulation):
    """Class to slice inputs by a specified index tag
    """
    name = "slice"

    tag = None

    def __init__(self, config):
        self.config = config
        Manipulation.__init__(self)
        self.tag = config.tag

    def set_component_outputs(self):
        # make datapack
        dp = DataPack(self.output)
        dp.add_usage(self.output_idx)
        self.data_pool.add_data_packs([dp], self.name)

    def produce_outputs(self):
        # fetch indices for the slice tag
        indices = self.input_dps.get_usage(Indices)
        indices_with_tag = indices.get_tag_instances(self.tag)
        info(f"Slicing with tag: [{self.tag}] indexes of size: {len(indices_with_tag)}")
        sliced_instances = self.input_dps.data.get_slice(indices_with_tag)
        sliced_index = np.arange(len(sliced_instances))

        # filter indices that fall within the selection and make Index obj
        other_idxs, other_tags = indices.get_overlapping(indices_with_tag, self.tag)
        self.output_idx = Indices([sliced_index], self.tag)
        for ix, tg in zip(other_idxs, other_tags):
            # re-align other indexes
            ix = np.concatenate([np.where(indices_with_tag == i)[0] for i in ix])
            self.output_idx.add_instance(ix, tg)
        # get sliced data
        data_cls = get_data_class(self.input_dps.data)
        # make output data
        self.output = data_cls(sliced_instances)

    def get_component_inputs(self):
        self.inputs = []
        self.indices = []
        # inputs = self.data_pool.request_data(Numeric, usage=Indices, usage_matching="subset", client=self.name, must_be_single=False)
        self.input_dps = self.data_pool.request_data(None, usage=Indices, usage_matching="subset", client=self.name, must_be_single=False)
        error(f"{self.name} can only operate on a single input", len(self.inputs) > 1)
        self.input_dps = self.input_dps[0]

# from manip.filter import Filter
from manip.manip import Manipulation
from utils import info, error, is_collection, debug, as_list
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
        self.target_tags = self.config.target_tags
        if self.target_tags is not None:
            self.target_tags = as_list(self.target_tags)

    def set_component_outputs(self):
        # make datapack
        # dp = DataPack(self.output)
        # dp.add_usage(self.output_idx)
        self.data_pool.add_data_packs([self.output_dp], self.name)

    def slice_dummy(self):
        # if the data to slice is dummy, slice the accompanying usages instead
        index_usages = self.input_dp.get_usage(Indices, allow_multiple=True)
        output_usages = []
        for index_usage in index_usages:
            idxs, tags = [], []
            for inst, tg in zip(index_usage.instances, index_usage.tags):
                if self.target_tags is not None and tg not in self.target_tags:
                    info(f"Skipping tag {tg} due to specified target tags: {self.target_tags}")
                    continue
                if self.config.rename_tag is not None and tg in self.config.rename_tag:
                        newtag = self.config.rename_tag[tg]
                        info(f"Renaming tag [{tg}] -> [{newtag}] in usage slicing")
                        tg = newtag
                sliced = inst[self.tagged_idx]
                idxs.append(sliced)
                tags.append(tg)
            usg = type(index_usage)(idxs, tags, skip_empty=False)
            output_usages.append(usg)

        self.output_dp = DataPack(DummyData(), output_usages)

    #     return self.input_dps.data.get_slice(self.tagged_idx)
    def produce_outputs(self):
        sliced_index = np.arange(len(self.tagged_idx))
        # fetch indices for the slice tag
        if type(self.input_dp.data) is DummyData:
            return self.slice_dummy()

        # make the data
        sliced_instances = self.input_dp.data.get_slice(self.tagged_idx)
        data_cls = get_data_class(self.input_dp.data)
        data = data_cls(sliced_instances)

        info(f"Slicing with tag: [{self.tag}] indexes of size: {len(self.tagged_idx)}")

        # make output datapack
        self.output_dp = DataPack(data)
        # make the sliced indices object
        output_idx = Indices([sliced_index], self.tag)

        # slice accompanying index usages
        index_usages = self.input_dp.get_usage(Indices, allow_multiple=True)
        for index_usage in index_usages:
            # filter indices that fall within the selection and make Index obj
            other_idxs, other_tags = index_usage.get_overlapping(self.tagged_idx, self.tag)
            # add the other indices
            for ix, tg in zip(other_idxs, other_tags):
                if self.target_tags is not None and tg not in self.target_tags:
                    info(f"Skipping tag {tg} because specified target tags: {self.target_tags}")
                    continue
                # re-align the other index wrt. the slicing
                ix = np.concatenate([np.where(self.tagged_idx == i)[0] for i in ix])
                # add the idx to the indices object
                output_idx.add_instance(ix, tg)

        # add the usage to the datapack
        self.output_dp.add_usage(output_idx)

    def get_component_inputs(self):
        """
        Slicing can be parameterized by a tag. That tag may reside:
        a) in data to be sliced itself
        b) in a separate datapack, of data type "dummy"
        """
        self.inputs, self.indices = [], []
        dummy_data_with_tag = None
        input_dps = self.data_pool.get_current_inputs()

        # fetch indices with the tag to slice with
        # if there's input data with the slicing tag, isolate them 
        dps_with_tag = defaultdict(list)

        for dp in input_dps:
            usgs = dp.get_usage(Indices, allow_multiple=True, allow_superclasses=True)
            for u in usgs:
                if u.has_tag(self.tag):
                    tag_idxs = u.get_tag_instances(self.tag, must_exist=False)
                    dps_with_tag[dp].append(tag_idxs)

        # ensure only one
        if len(dps_with_tag) > 1:
            error(f"Slicing tag {self.tag} found in more than one input datapacks: {dps_with_tag.keys()}!")
        multiple_instances = [dp for dp, v in (dps_with_tag.items()) if len(v) > 1]
        if multiple_instances:
            error(f"Slicing tag {self.tag} found in more than one index usage in datapack {multiple_instances}")
        if  len(dps_with_tag) == 0:
            error(f"No datapack found with slicing tag [{self.tag}]")
        tag_dp, tag_idx =  dps_with_tag.popitem()
        self.tagged_idx = tag_idx[0]
        is_dummy = type(tag_dp.data) is DummyData

        if is_dummy:
            # exclude the slicing dp from slicing, if it's a dummy
            # since it contains no other data / indexes to slice
            input_dps = [x for x in input_dps if x != tag_dp]
        # single-dp
        error(f"Invalid number of datapacks to slice: {len(input_dps)}", len(input_dps) != 1)
        self.input_dp = input_dps[0]

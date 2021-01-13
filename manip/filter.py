from manip.manip import Manipulation
from utils import shapes_list, error, data_summary, debug, info, to_namedtuple
from bundle.datausages import *
from bundle.datatypes import *
import numpy as np

"""
Vector concatenation manipulation component
"""
class Filter(Manipulation):
    # column-wise concatenation
    name = "filter"
    func = None

    def __init__(self, config):
        Manipulation.__init__(self)
        self.config = config
        self.params = config.params
        try:
            self.func = eval(config.function)
            info(f"Using filter function: {str(config.function)}")


        except KeyError:
            error(f"Undefined filtering predefined function: {config.function}")

    def apply_operation(self, inputs):
        if self.config.params is not None:
            return self.func(inputs, self.params)
        return self.func(inputs)

    def produce_outputs(self):
        self.outputs = []
        info(f"Applying filter: {self.config.function}")
        if self.config.params is not None:
            debug(f"Using params: {str(self.params)}")
        for i, dp in enumerate(self.input_dps):
            info(f"Applying filter to datapack {i+1}/{len(self.input_dps)} of len/shp: {dp.data.get_shape_info()}")
            instances = dp.data.instances
            if len(instances) == 0:
                mask = np.empty(0, dtype=np.int64)
            else:
                mask = self.apply_operation(instances)
            debug(f"First 10 of the {len(mask)} survivors: {mask[:10]}")
            # add filtered index and tag
            output_usages = [Indices(mask, [self.config.produce_index_tag], skip_empty=False)]
            # modify (align) existing indexes, post-filtering
            for u in dp.usages:
                if issubclass(type(u), Indices):
                    u = u.apply_mask(mask)
                output_usages.append(u)

            # filtering output -- dummy data
            new_dp = DataPack(DummyData(), output_usages)
            self.outputs.append(new_dp)

    def set_component_outputs(self):
        self.data_pool.add_data_packs(self.outputs, self.name)

    def get_component_inputs(self):
        # filter component can get any type inputs fed to it
        self.input_dps = self.data_pool.get_current_inputs()
        error(f"No inputs available for {self.name}", len(self.input_dps) == 0)
        # get config
        if self.config.params == "input":
            self.params = self.data_pool.request_data(Dictionary, usage="ignore", client=self.name, reference_data=self.data_pool.data)
            self.params = to_namedtuple(self.params.data.instances, "params")
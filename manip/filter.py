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
    axis = 1

    func = None

    def __init__(self, config):
        Manipulation.__init__(self)
        self.config = config
        self.params = config.params
        self.alters_index = config.alters_index
        try:
            self.func = eval(config.function)
        except KeyError:
            error(f"Undefined filtering predefined function: {config.function}")

    def apply_operation(self, inputs):
        if self.config.params is not None:
            return self.func(inputs, self.params)
        return self.func(inputs)

    def produce_outputs(self):
        self.outputs = []
        info(f"Applying filter: {self.func}")
        for i, dp in enumerate(self.input_dps):
            func_output = self.apply_operation(dp.data.instances)
            # usage handling
            if self.alters_index:
                # get post-filtering indexes
                mask = func_output
                output_usages = []
                for u in dp.usages:
                    if issubclass(type(u), Indices):
                        u = u.apply_mask(mask)
                    output_usages.append(u)
                output_instances = dp.data.get_slice(mask)
            else:
                output_instances = func_output
                output_usages = dp.usages

            data_cls = get_data_class(dp.data)
            new_dp = DataPack(data_cls(output_instances), output_usages)
            self.outputs.append(new_dp)
            info(f"Filtering {i+1}/{len(self.input_dps)} input data pack")

    def set_component_outputs(self):
        self.data_pool.add_data_packs(self.outputs, self.name)

    def get_component_inputs(self):
        # filter component can get any type inputs fed to it
        self.input_dps = self.data_pool.get_current_inputs()
        # get config
        if self.params == "input":
            self.params = self.data_pool.request_data(Dictionary, usage="ignore", client=self.name, reference_data=self.data_pool.data)
            self.params = to_namedtuple(self.params.data.instances, "params")
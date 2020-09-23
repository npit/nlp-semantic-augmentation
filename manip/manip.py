"""Vector manipulation"""
from bundle.bundle import DataPool
from bundle.datatypes import Numeric
from component.component import Component
from utils import error
import numpy as np


class Manipulation(Component):
    component_name = "manipulation"

    produces=Numeric
    consumes=Numeric
    def __init__(self):
        pass

    def configure_name(self):
        self.source_name = self.inputs.get_source_name()
        if type(self.source_name) in [list, tuple]:
            self.source_name = "_".join(self.source_name)
        self.name = "{}_{}".format(self.source_name, self.name)
        Component.configure_name(self, self.name)

    def process_component_inputs(self):
        # if len(self.inputs) > 1:
        #     self.vectors = [x.instances for x in self.inputs.get_vectors()]
        #     for v, vecs in enumerate(self.vectors):
        #         if vecs is None:
        #             bund = self.inputs.get(v)
        #             error("Specified {} manip for chain {} / bundle {}, which does not contain vectors.".format(
        #                 self.name, bund.get_chain_name(), bund.get_source_name()))
        # else:
        #     self.vectors = self.inputs.get_vectors().instances
        self.vectors = [x.get_vectors() for x in self.inputs.get_vectors(full_search=True, enforce_single=False)]
        self.indices = [x.get_indices() for x in self.inputs.get_indices(full_search=True, enforce_single=False)]

        # check that all index instances are equal across all input bundles that contain Indeces objects
        for inst_idx in range(len(self.indices[0].instances)):
            for ind_obj_idx in range(0, len(self.indices)-1):
                inst1 = self.indices[ind_obj_idx].instances[inst_idx]
                inst2 = self.indices[ind_obj_idx+1].instances[inst_idx]
                if not np.array_equal(inst1, inst2):
                    error(f"Unequal input indeces in {self.name} component: {inst1} {inst2}")
        # all are equal, set the first
        self.indices = self.indices[0]
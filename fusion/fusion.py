from bundle.bundle import Bundle, BundleList
from bundle.datatypes import Vectors
from component.component import Component
from utils import error, shapes_list, info, warning
import numpy as np
"""Class that implements the combination of two chains"""


class Fusion(Component):
    component_name = "fusion"

    def __init__(self, config):
        Component.__init__(self, consumes=Vectors.name, produces=Vectors.name)
        pass
        # self.num_inputs = len(config.fusion.inputs)

    def load_inputs(self, inputs):
        self.inputs = inputs

    def fuse(self):
        error("Attempted to call abstract fuse() from component {}".format(self.name))

    def run(self):
        self.process_component_inputs()

        # fuse existing vectors
        self.vectors = list(filter(lambda x: x is not None, self.vectors))
        num_vector_collections = list(map(len, self.vectors))
        error("Inconsistent number of vector collections to fuse: {}".format(num_vector_collections), len(set(num_vector_collections)) != 1)
        self.vectors = list(zip(*self.vectors))
        for v, vecs in enumerate(self.vectors):
            msg = "Fusing input collection {}/{} with shapes: {} to".format(v + 1, len(self.vectors), shapes_list(vecs))
            self.vectors[v] = self.fuse(vecs)
            info(msg + " {}".format(self.vectors[v].shape))

        self.outputs.set_vectors(Vectors(vecs=self.vectors))

    def process_component_inputs(self):
        # make sure input is a collection of bundles
        self.vectors = [x.get_instances() for x in self.inputs.get_vectors()]
        # epis = [x.elements_per_instance for x in self.inputs.get_vectors()]
        # error("Unequal elements per instance during {} fusion".format(str(epis)), np.all(np.diff(epis)) == 0)
        error("{} can only fuse a collection of bundles".format(self.name), type(self.inputs) is not BundleList)
        for v, vecs in enumerate(self.vectors):
            if vecs is None:
                bund = self.inputs.get(v)
                error("Specified {} fusion for chain {} / bundle {}, which does not contain vectors.".format(
                    self.name, bund.get_chain_name(), bund.get_source_name()))

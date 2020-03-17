import numpy as np

from bundle.bundle import Bundle
from bundle.datatypes import Vectors
from manip.manip import Manipulation
from utils import error, info, shapes_list, warning

"""Class that implements the combination of two chains"""


class Fusion(Manipulation):

    def __init__(self, _):
        Manipulation.__init__(self)
        # self.num_inputs = len(config.manip.inputs)

    def fuse(self):
        error("Attempted to call abstract fuse() from component {}".format(self.name))

    def run(self):
        self.process_component_inputs()

        # fuse existing vectors
        self.vectors = list(filter(lambda x: x is not None, self.vectors))
        num_vector_collections = list(map(len, self.vectors))
        error("Inconsistent number of vector collections to fuse: {}".format(num_vector_collections), len(set(num_vector_collections)) != 1)
        error("Inconsistent number of vector collections to fuse: {}".format(num_vector_collections), len(set(num_vector_collections)) != 1)
        self.vectors = list(zip(*self.vectors))
        for v, vecs in enumerate(self.vectors):
            msg = "Fusing input collection {}/{} with shapes: {} to".format(v + 1, len(self.vectors), shapes_list(vecs))
            self.vectors[v] = self.fuse(vecs)
            info(msg + " {}".format(self.vectors[v].shape))

        self.outputs.set_vectors(Vectors(vecs=self.vectors, epi=[np.ones(len(vecs), np.int32) * self.num_elements_per_instance for vecs in self.vectors]))

    def process_component_inputs(self):
        # make sure input is a collection of bundles
        Manipulation.process_component_inputs(self)
        all_epis = np.concatenate(np.concatenate([x.elements_per_instance for x in self.inputs.get_vectors()]))
        all_epis = np.unique(all_epis)
        error("Cannot fuse inputs with different elements per instance: {}".format(all_epis), len(all_epis) > 1)
        self.num_elements_per_instance = all_epis[0]

        # error("Unequal elements per instance during {} manip".format(str(epis)), np.all(np.diff(epis)) == 0)
        error("{} can only fuse a collection of bundles".format(self.name), len(self.inputs) <= 1)
        for v, vecs in enumerate(self.vectors):
            if vecs is None:
                bund = self.inputs.get(v)
                error("Specified {} manip for chain {} / bundle {}, which does not contain vectors.".format(
                    self.name, bund.get_chain_name(), bund.get_source_name()))

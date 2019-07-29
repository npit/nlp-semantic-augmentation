"""Vector manipulation"""
from bundle.bundle import BundleList
from bundle.datatypes import Vectors
from component.component import Component
from utils import error

class Manipulation(Component):
    def __init__(self):
        Component.__init__(self, produces=Vectors, consumes=Vectors)

    def process_component_inputs(self):
        if type(self.inputs) is BundleList:
            self.vectors = [x.get_instances() for x in self.inputs.get_vectors()]
            for v, vecs in enumerate(self.vectors):
                if vecs is None:
                    bund = self.inputs.get(v)
                    error("Specified {} manip for chain {} / bundle {}, which does not contain vectors.".format(
                        self.name, bund.get_chain_name(), bund.get_source_name()))
        else:
            self.vectors = self.inputs.get_vectors().instances

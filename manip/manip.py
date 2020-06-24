"""Vector manipulation"""
from bundle.bundle import Bundle
from bundle.datatypes import Vectors
from component.component import Component
from utils import error


class Manipulation(Component):
    component_name = "manipulation"
    def __init__(self):
        Component.__init__(self, produces=Vectors, consumes=Vectors)

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
        self.vectors = [x.instances for x in self.inputs.get_vectors()]

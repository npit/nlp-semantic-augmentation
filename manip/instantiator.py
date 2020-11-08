from manip.ngram import NGram
from manip.slice import Slice
from manip.concat import Concatenation
from manip.replication import Replication
from manip.filter import Filter
from utils import error

class Instantiator:
    component_name = "manip"
    candidates = [Concatenation, Replication, Filter, NGram, Slice]

    @staticmethod
    def create(config):
        for cand in Instantiator.candidates:
            if cand.name == config.name:
                return cand(config)
        error("Undefined {} : {}".format(Instantiator.component_name, config.name))

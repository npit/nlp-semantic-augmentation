from manip.concat import Concatenation
from manip.replication import Replication
from utils import error

class Instantiator:
    component_name = "manip"
    def create(config):
        if config.name == Concatenation.name:
            return Concatenation(config)
        if config.name == Replication.name:
            return Replication(config)
        error("Undefined {} : {}".format(Instantiator.name, config.manip.name))

from fusion.concat import Concatenation
from utils import error

class Instantiator:
    component_name = "fusion"
    def create(config):
        if config.fusion.name == "concat":
            return Concatenation(config)
        error("Undefined {} : {}".format(Instantiator.name, config.fusion.name))

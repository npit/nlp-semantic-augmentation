from endpoint.instantiator import Instantiator as endpoint_instantiator
from component.trigger import ImmediateExecution
from utils import error

class TriggerInstantiator:
    @staticmethod
    def create(name, conf):
        if name == endpoint_instantiator.component_name:
            return endpoint_instantiator.create(conf)
        error(f"Undefined trigger: {name}")

    @staticmethod
    def make_default(conf):
        return ImmediateExecution(conf)
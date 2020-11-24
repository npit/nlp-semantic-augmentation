from component.trigger import ImmediateExecution
from endpoint.endpoint import IOEndpoint
from utils import error

class TriggerInstantiator:
    @staticmethod
    def create(trigger_name, conf):
        candidates = [IOEndpoint]
        name = conf.name
        for c in candidates:
            if name == c.name:
                return c(trigger_name, conf)
        error(f"Undefined trigger type {name} for trigger: {trigger_name}. Candidates are: {[c.name for c in candidates]}")

    @staticmethod
    def make_default(conf):
        return ImmediateExecution("immediate", conf)

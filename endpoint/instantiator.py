"""Endpoint instantiation module"""
from utils import error
from endpoint.endpoint import IOEndpoint

class Instantiator:
    """Class to instantiate a dataset object"""
    component_name = "endpoint"

    @staticmethod
    def create(config):
        name = config.name
        if name == IOEndpoint.name:
            return IOEndpoint(config)
        error(f"Undefined endpoint name {name}")

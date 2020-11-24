"""Endpoint instantiation module"""
from utils import error
from endpoint.endpoint import IOEndpoint

class Instantiator:
    """Class to instantiate an endpoint object"""
    component_name = "rest"

    @staticmethod
    def create(config):
        name = config.name
        if name == IOEndpoint.name:
            return IOEndpoint(config)
        error(f"Undefined endpoint name {name}")

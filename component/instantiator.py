from dataset.instantiator import Instantiator as dset_instantiator
from representation.instantiator import Instantiator as rep_instantiator
from semantic.instantiator import Instantiator as sem_instantiator
from transform.instantiator import Instantiator as tra_instantiator
from learning.instantiator import Instantiator as lrn_instantiator
from fusion.instantiator import Instantiator as fus_instantiator
from component.link import Link
from utils import error

"""Generic component instantiator"""


def create(component_name, component_params):
    if component_name == dset_instantiator.name:
        return dset_instantiator.create(component_params)
    if component_name == rep_instantiator.name:
        return rep_instantiator.create(component_params)
    if component_name == sem_instantiator.name:
        return sem_instantiator.create(component_params)
    if component_name == tra_instantiator.name:
        return tra_instantiator.create(component_params)
    if component_name == lrn_instantiator.name:
        return lrn_instantiator.create(component_params)
    if component_name == fus_instantiator.name:
        return fus_instantiator.create(component_params)
    if component_name == Link.name:
        return Link(component_params)
    error("Undefined component type: {}".format(component_name))

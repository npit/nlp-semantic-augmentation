from dataset.instantiator import Instantiator as dset_instantiator
from representation.instantiator import Instantiator as rep_instantiator
from semantic.instantiator import Instantiator as sem_instantiator
from transform.instantiator import Instantiator as tra_instantiator
from learning.instantiator import Instantiator as lrn_instantiator
from manip.instantiator import Instantiator as manip_instantiator
from evaluation.instantiator import Instantiator as eval_instantiator
from sampling.sampling import Instantiator as smpl_instantiator
from report.instantiator import Instantiator as report_instantiator

from utils import error

"""Generic component instantiator"""


def create(component_name, component_params):
    if component_name == dset_instantiator.component_name:
        return dset_instantiator.create(component_params)
    if component_name == rep_instantiator.component_name:
        return rep_instantiator.create(component_params)
    if component_name == sem_instantiator.component_name:
        return sem_instantiator.create(component_params)
    if component_name == tra_instantiator.component_name:
        return tra_instantiator.create(component_params)
    if component_name == lrn_instantiator.component_name:
        return lrn_instantiator.create(component_params)
    if component_name == manip_instantiator.component_name:
        return manip_instantiator.create(component_params)
    if component_name == eval_instantiator.component_name:
        return eval_instantiator.create(component_params)
    if component_name == smpl_instantiator.component_name:
        return smpl_instantiator.create(component_params)
    if component_name == report_instantiator.component_name:
        return report_instantiator.create(component_params)
    error("Undefined component type: {}".format(component_name))

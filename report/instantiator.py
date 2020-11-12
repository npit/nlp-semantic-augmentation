"""Report instantiation module"""
from utils import error
from report.report import MultistageClassificationReport

class Instantiator:
    """Class to instantiate a report object"""
    component_name = "report"

    @staticmethod
    def create(config):
        name = config.name
        if name == MultistageClassificationReport.name:
            return MultistageClassificationReport(config)
        error(f"Undefined report name {name}")

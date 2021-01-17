"""Report instantiation module"""
from utils import error
from report.report import MultistageClassificationReport
from report.nvreport import NVReport

class Instantiator:
    """Class to instantiate a report object"""
    component_name = "report"

    @staticmethod
    def create(config):
        name = config.name
        if name == MultistageClassificationReport.name:
            return MultistageClassificationReport(config)
        if name == NVReport.name:
            return NVReport(config)
        error(f"Undefined report name {name}")

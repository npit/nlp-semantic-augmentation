from scipy.stats import shapiro

from stattests.stattest import StatisticalTest


class GaussianityTest(StatisticalTest):
    pass

class ShapiroWilk(GaussianityTest):
    name = "shapirowilk"
    def run(self, data):
        """Run the shapiro-wilk gaussianness test. Assumes IID data."""
        self.statistic, self.p_value = shapiro(data)

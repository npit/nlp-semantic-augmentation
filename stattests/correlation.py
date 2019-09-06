
from scipy.stats import kendalltau, pearsonr, spearmanr

from stattests.stattest import StatisticalTest


class CorrelationTest(StatisticalTest):
    pass

class Pearson(CorrelationTest):
    name = "pearson"
    def run(self, data1, data2):
        """Tests for linear correlation relationship. Assumes IID, normality, same variance."""
        self.statistic, self.p_value = pearsonr(data1, data2)

class Spearman(CorrelationTest):
    name = "spearman"
    def run(self, data1, data2):
        """Tests for non-linear correlation relationship (monotonicity). Assumes IID, and an order relation applicable in the data."""
        self.statistic, self.p_value = pearsonr(data1, data2)

class Kendall(CorrelationTest):
    name = "kendall"
    def run(self, data1, data2):
        """Tests for non-linear correlation relationship (monotonicity). Assumes IID, and an order relation applicable in the data."""
        self.statistic, self.p_value = kendalltau(data1, data2)

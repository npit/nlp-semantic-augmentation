"""Statistical significance differene tests."""
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd

from stattests.stattest import StatisticalTest


class Difference(StatisticalTest):
    pass

class Anova(Difference):
    name = "anova"
    """Assumes normality, IID."""
    def run(self, data1, data2):
        self.statistic, self.p_value = f_oneway(data1, data2)

class TukeyHSD(Difference):
    name = "tukeyhsd"
    # from https://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/
    """Assumes normality, IID, one-to-one data pairing."""
    def run(self, data, groups):
        mc = MultiComparison(data, groups)
        result = mc.tukeyhsd()
        return result.__str__(), mc.groupsunique
        # print(result)
        # print(mc.groupsunique)

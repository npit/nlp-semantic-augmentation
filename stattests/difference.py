"""Statistical significance differene tests."""
from itertools import combinations

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd

from stattests.stattest import StatisticalTest
from utils import info


class Difference(StatisticalTest):
    pass

class Anova(Difference):
    name = "anova"
    """Assumes normality, IID."""
    def run(self, data, groups):
        self.results = []
        for name1, name2 in combinations(set(groups), 2):
            # get data per pairwise comp
            data1, data2 = [[data.iloc[i] for i in range(len(groups)) if groups.iloc[i] == n]
                            for n in (name1, name2)]
            stat, p = f_oneway(data1, data2)
            self.results.append((stat, p, (name1, name2)))

    def report(self):
        info("{} results:".format(self.name))
        for res in self.results:
            info("F-stat: {:.3f}, p-value: {:.3f}, comparison: {}".format(*res))

class TukeyHSD(Difference):
    name = "tukeyhsd"
    # from https://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/
    """Assumes normality, IID, one-to-one data pairing."""
    def run(self, data, groups):
        self.mc = MultiComparison(data, groups)
        self.result = self.mc.tukeyhsd()
        return self.result, self.mc.groupsunique

    def report(self):
        info("{} results:".format(self.name))
        info("result: {}".format(self.result.__str__()))
        info("Unique groups: {}".format(self.mc.groupsunique))

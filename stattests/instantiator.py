from stattests import difference, correlation, gaussianity
from utils import error

class Instantiator:

    def create(self, name):
        """Function to instantiate a statistical test object"""
        candidates = [gaussianity.ShapiroWilk, \
                      difference.Anova, difference.TukeyHSD, \
                      correlation.Kendall, correlation.Pearson, correlation.Spearman ]
        for candidate in candidates:
            if name == candidate.name:
                return candidate()
        error("Undefined statistical test: {}".format(name))


# determine abstract classes for the generic types of statistical tests

# most from https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
class StatisticalTest:
    def report(self):
        """Report test results"""
        error("Attempted to access method report() function for {}.".format(self.name))

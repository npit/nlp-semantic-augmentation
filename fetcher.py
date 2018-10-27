from dataset import TwentyNewsGroups, Reuters
from embedding import Glove, Word2vec, Train
from semantic import Wordnet
from utils import error
from learner import MLP, LSTM


class Fetcher:
    # split name from params
    def get_params(self, composite_str):
        # format expected is name,param1,param2
        nameparams = composite_str.split(",")
        name, params = nameparams[0], nameparams[1:]
        return name, params

    # datasets name resolver
    def fetch_dataset(self, name):
        name, params = self.get_params(name)
        if name == TwentyNewsGroups.name:
            return TwentyNewsGroups()
        elif name == Reuters.name:
            return Reuters()
        else:
            error("Undefined dataset: {}".format(name))

    # semantic name resolver
    def fetch_semantic(self, name):
        name, params = self.get_params(name)
        if name == Wordnet.name:
            return Wordnet()
        else:
            error("Undefined semantic resource: {}".format(name))



from dataset import TwentyNewsGroups
from embedding import Glove, Word2vec
from semantic import Wordnet
from helpers import error
from learner import Dnn


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
        else:
            error("Undefined dataset: {}".format(name))

    # semantic name resolver
    def fetch_semantic(self, name):
        name, params = self.get_params(name)
        if name == Wordnet.name:
            return Wordnet()
        else:
            error("Undefined semantic resource: {}".format(name))

    # embedding name resolver
    def fetch_embedding(self, name):
        name, params = self.get_params(name)
        if name == Glove.name:
            return Glove(params)
        if name == Word2vec.name:
            return Word2vec(params)
        else:
            error("Undefined embedding: {}".format(name))


    # embedding name resolver
    def fetch_learner(self, name):
        name, params = self.get_params(name)
        if name == Dnn.name:
            return Dnn(params)
        else:
            error("Undefined embedding: {}".format(name))

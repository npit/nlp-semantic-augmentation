from semantic.babelnet import BabelNet
# from semantic.context_embedding import ContextEmbedding
from semantic.dbpedia import DBPedia
from semantic.framenet import Framenet
from semantic.google_knowledge_graph import GoogleKnowledgeGraph
from semantic.wordnet import Wordnet
from utils import error


class Instantiator:
    component_name = "semantic"

    def create(config):
        name = config.name
        if name == Wordnet.name:
            return Wordnet(config)
        if name == GoogleKnowledgeGraph.name:
            return GoogleKnowledgeGraph(config)
        # if name == ContextEmbedding.name:
        #     return ContextEmbedding(config)
        if name == Framenet.name:
            return Framenet(config)
        if name == BabelNet.name:
            return BabelNet(config)
        if name == DBPedia.name:
            return DBPedia(config)
        error("Undefined semantic resource: {}".format(name))

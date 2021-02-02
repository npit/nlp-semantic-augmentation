from semantic.semantic_resource import SemanticResource


class BabelNet(SemanticResource):
    name = "babelnet"

    def get_raw_path(self):
        return None

    def __init__(self, config):
        self.config = config
        SemanticResource.__init__(self)
        # map nltk pos maps into meaningful framenet ones
        self.pos_tag_mapping = {"VB": "V", "NN": "N", "JJ": "A", "RB": "ADV"}

    # lookup for babelnet should be about a (large) set of words
    # written into a file, read by the java api
    # results written into a file (json), read from here.
    # run calls the java program


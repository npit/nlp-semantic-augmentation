from semantic.semantic_resource import SemanticResource
from utils import error
from os.path import exists


class DBPedia(SemanticResource):
    """Semantic extractor from a dbpedia local docker container

    Provide a dbpedia-docker.yml with the following entries:
    # the host:port of the docker container
    rest_url: localhost:port
    # the confidence level of the semantic extraction
    confidence: <num>
    """

    dbpedia_config = "dbpedia-docker.yml"
    name = "dbpedia"

    def lookup(self, candidate):
        concepts_scores = {}
        try:
            # ret example:
            # {'URI': 'http://dbpedia.org/resource/Dog', 'similarityScore': 0.9997269051125752, 'offset': 10, 'percentageOfSecondRank': 0.0002642731468138545, 'support': 12528, 'types': '', 'surfaceForm': 'dog'}
            # {'URI': 'http://dbpedia.org/resource/Cat', 'similarityScore': 0.9972015232913088, 'offset': 14, 'percentageOfSecondRank': 0.002500864314071041, 'support': 7471, 'types': '', 'surfaceForm': 'cat'}]
            res = spotlight.annotate(self.rest_url, candidate, confidence=self.confidence)
            for element in res:
                name, score = element["URI"], element["similarityScore"]
                # only update if a better similarity score is detected
                if name in concepts_scores and concepts_scores[name] <= score:
                    continue
                concepts_scores[name] = score
                # debug("Got URI {} | score: {}".format(name, score))
        except spotlight.SpotlightException as se:
            debug(se)
        except requests.exceptions.HTTPError as ex:
            debug(ex)
        return concepts_scores

    def get_term_delineation(self, document_text):
        """ Function to produce a list of terms of interest, from which to extract concepts """
        # DBPedia extracts concepts directly from the full text
        return [" ".join([x[0] for x in document_text])]

    def __init__(self, config):
        self.do_cache = False
        self.config = config

        # dbpedia conf
        if not exists(self.dbpedia_config):
            error("The {} semantic resource needs an extractor configuration file: {}".format(self.name, self.dbpedia_config))
        with open(self.dbpedia_config) as f:
            dbpedia_conf = yaml.load(f, Loader=yaml.SafeLoader)
        self.rest_url = dbpedia_conf["rest_url"]
        self.confidence = dbpedia_conf["confidence"]
        SemanticResource.__init__(self)

    # use pyspotlight package
    # get url of docker rest api from the config
    # however this class processed document text as a whole, not word for word
    # need to rework

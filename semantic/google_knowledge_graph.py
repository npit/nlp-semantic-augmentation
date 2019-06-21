import json
import urllib

from semantic.semantic_resource import SemanticResource


class GoogleKnowledgeGraph(SemanticResource):
    name = "googlekt"

    query_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    key = None

    def __init__(self, config):
        self.config = config
        self.key = config.misc.keys["googleapi"]
        self.query_params = {
            'limit': 10,
            'indent': True,
            'key': self.key,
        }
        SemanticResource.__init__(self)

    def lookup(self, candidate):
        word, pos_info = candidate
        self.query_params["query"] = word
        url = self.query_url + '?' + urllib.parse.urlencode(self.query_params)
        response = json.loads(urllib.request.urlopen(url).read())
        names, hypers, scores = [], [], []
        for element in response['itemListElement']:
            results = element['result']
            if "name" not in results:
                continue
            scores.append(element['resultScore'])
            names.append(results['name'])
            hypers.append(results['@type'])
            # descr = results['description']
            # detailed_descr = results['detailedDescription'] if 'detailedDescription' in results else None

        names = self.disambiguate(names, candidate)
        activations = {n: 1 for n in names}

        if self.do_spread_activation:
            for name in names:
                idx = names.index(name)
                hyps = hypers[idx]
                activations[name] = 1
                if self.do_spread_activation:
                    current_decay = self.spread_decay_factor
                    for h, hyp in enumerate(hyps):
                        if h + 1 > self.spread_steps:
                            break
                        activations[hyp] = current_decay
                        current_decay *= self.spread_decay_factor
        return activations


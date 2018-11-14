
"""
Definitions file, serving the role of hierarchical constants.
"""
def avail(cls):
    ret = [v for (k,v) in dict(vars(cls)).items()  if not k.startswith("__") and not callable(v)] 
    return ret

class semantic:

    class disam:
        first, pos = "first", "pos"
    class weights:
        frequencies, tfidf = "frequencies", "tfidf" 
        def avail():
            return avail(semantic.weights)
        def to_string(config):
            return "w{}".format(config.semantic.weights)

    class limit:
        frequency, top, none = "frequency", "top", "all"
        def avail():
            return avail(semantic.limit)
        def to_string(config=None, value=None):
            if value is None:
                value = config.semantic.limit
            return "ALL" if  value is None else "".join(list(map(str, value)))

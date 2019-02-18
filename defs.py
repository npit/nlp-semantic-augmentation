
"""
Definitions file, serving the role of hierarchical constants.
"""


def avail(cls):
    ret = [v for (k,v) in dict(vars(cls)).items()  if not k.startswith("__") and not callable(v)] 
    return ret


class weights:
    frequencies, tfidf = "frequencies", "tfidf"

    def avail():
        return avail(weights)

    def to_string(w):
        return "w{}".format(w)


class semantic:

    class disam:
        first, pos = "first", "pos"


class limit:
    frequency, top, none = "frequency", "top", "all"

    def avail():
        return avail(limit)

    def to_string(value):
        return "ALL" if value is None else "".join(list(map(str, value)))

class alias:
    none = "none"

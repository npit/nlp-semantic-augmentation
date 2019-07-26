from utils import to_namedtuple
"""
Definitions file, serving the role of hierarchical constants.
"""


def make_def(name):
    avail_list = eval("avail_" + name)
    return to_namedtuple(ntname=name, conf_dict= {k:k if k != "avail" else avail_list for k in avail_list + ["avail"]})


avail_aggregation = ["pad", "avg"]
avail_sequence_length = [ "unit", "non_unit"]
avail_weights = ["frequencies", "tfidf"]
avail_disam = ["first", "pos"]
avail_limit = ["frequency", "top", "none"]
avail_alias = ["none", "link"]
avail_sampling = ["oversample", "undersample"]

aggregation = make_def("aggregation")
sequence_length = make_def("sequence_length")
disam = make_def("disam")
limit = make_def("limit")
alias = make_def("alias")
weights = make_def("weights")
sampling = make_def("sampling")

def is_none(elem):
    return elem == '' or elem is None or elem == alias.none or not elem

def get_sequence_length_type(inp):
    if inp == 1:
        return sequence_length.unit
    return sequence_length.non_unit

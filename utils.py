import logging
import os
import pickle
import time
from collections import Counter, OrderedDict, namedtuple
from os.path import exists

import nltk
import numpy as np
import yaml

num_warnings = 0

def realign_embedding_index(data_indexes, all_indexes):
    """Check for non-mapped indexes, drop them and realign"""
    mapped_indexes = np.asarray([False for _ in all_indexes])
    for data_bundle in data_indexes:
        for indexes in data_bundle:
            mapped_indexes[indexes] = True
    # find unmapped
    new_indexes = all_indexes[mapped_indexes]
    old2new_index = {}
    for idx, oldidx in enumerate(new_indexes):
        old2new_index[oldidx] = idx

    # remap
    for d in range(len(data_indexes)):
        for i in range(len(data_indexes[d])):
            data_indexes[d][i] = [old2new_index[oldidx] for oldidx in data_indexes[d][i]]
    if len(all_indexes) < len(new_indexes):
        debug("Realignment reduced embedding matrix from {} to {} elements.".format(len(all_indexes), len(new_indexes)))
    return data_indexes, new_indexes

def match_labels_to_instances(elements_per_instance, labels):
    """Expand, if needed, ground truth samples for multi-vector instances
    """
    multi_vector_instance_idx = [i for i in range(len(elements_per_instance)) if elements_per_instance[i] > 1]
    if not multi_vector_instance_idx:
        return labels
    res = []
    for i in range(len(labels)):
        # get the number of elements for the instance
        times = elements_per_instance[i]
        res.extend([elements_per_instance[i] for _ in range(times)])
    return res


# check for zero length on variable input args
def zero_length(*args):
    for arg in args:
        if len(arg) == 0:
            return True
    return False


# convert a numeric or a set of numerics to string
def numeric_to_string(value, precision):
    try:
        # iterable scores values
        iter(value)
        try:
            # single iterables
            return "{" + " ".join(list(map(lambda x: precision.format(x), value))) + "}"
        except:
            # 2d iterable
            sc = []
            for k in value:
                sc.append("[{}]".format(" ".join(list(map(lambda x: precision.format(x), k)))))
            return "{" + " ".join(sc) + "}"
    except:
        return precision.format(value)


# for laconic passing to the error function
def ill_defined(var, can_be=None, cannot_be=None, func=None):
    return not well_defined(var, can_be, cannot_be, func)

# verifies that the input variable is either None, or conforms to acceptable (can only be such) or problematic (can not equal) values
def well_defined(var, can_be=None, cannot_be=None, func=None):
    if var is None:
        return True
    error("Specified both acceptable and problematic constraint values in well_defined", can_be is not None and cannot_be is not None)
    if func:
        return func(var)
    if can_be:
        return var == can_be
    return var != cannot_be

def nltk_download(config, name):
    nltk.download(name, download_dir=nltk.data.path[0])

def setup_simple_logging(level="info", logging_dir="."):
    level = logging._nameToLevel[level.upper()]
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    logfile = os.path.join(logging_dir, "experiments_{}.log".format(datetime_str()))

    chandler = logging.StreamHandler()
    chandler.setFormatter(formatter)
    chandler.setLevel(level)
    logging.getLogger().addHandler(chandler)

    fhandler = logging.FileHandler(logfile)
    fhandler.setLevel(level)
    fhandler.setFormatter(formatter)
    logging.getLogger().addHandler(fhandler)
    logging.getLogger().setLevel(logging.DEBUG)


def to_namedtuple(conf_dict, ntname, do_recurse=False):
    keys = sorted(conf_dict.keys())
    if do_recurse:
        # apply namedtuple conversion to internal dicts
        conf = namedtuple(ntname, keys)(*[to_namedtuple(conf_dict[k], "dummy") if type(conf_dict[k]) in [dict, OrderedDict] else conf_dict[k] for k in keys])
    else:
        conf = namedtuple(ntname, keys)(*[conf_dict[k] for k in keys])
    return conf

def get_type_name(data):
    return type(data).__name__

def as_list(x):
    """Convert the input to a single-element list, if it's not a list
    """
    return [x] if type(x) is not list else x

def is_multilabel(labels):
    try:
        # attempt to check length of iterate
        for sample_labels in labels:
            if len(sample_labels) > 1:
                return True
    except TypeError:
        pass
    return False

# function for one-hot encoding, can handle multilabel
def one_hot(labels, num_labels):
    output = np.empty((0, num_labels), np.float32)
    for annot in labels:
        binarized = np.zeros((1, num_labels), np.float32)
        if type(annot) is list:
            for lbl in annot:
                binarized[0, lbl] = 1.0
        else:
            binarized[0, annot] = 1.0
        output = np.append(output, binarized, axis=0)
    return output

def is_collection(data):
    return type(data) in [list, dict, OrderedDict, tuple]

def single_data_summary(data, data_index, recursion_depth=0):
    dtype = get_type_name(data)
    data_index = str(data_index) + ":"
    indent = recursion_depth * "  "
    msg = indent + "{}".format(data_index)
    if is_collection(data):
        msg += " {:15s}, {:10s}".format(str(len(data)) + " elements", dtype)
    elif type(data) is np.ndarray:
        msg += " {:15s} {:10s}".format(str(data.shape) + " shape", "ndarray")
    else:
        msg += "data of type {}".format(dtype)
    debug(msg)

def data_summary(data, msg="", data_index="", recursion_depth=0):
    """ Print a summary of data in the input"""
    coll_len_lim = 2
    recursion_depth_lim = 1
    if recursion_depth < recursion_depth_lim:
        recursion_depth += 1
        # recurse into collections
        if is_collection(data):
            colltype = type(data)
            if colltype in [dict, OrderedDict]:
                maxname = max(map(len, data.keys()))
                names = map(lambda x: ("{:" + str(maxname) + "s}").format(x), data.keys())
                data = [data[k] for k in names]
            else:
                names = range(1, len(data)+1)

            debug("{} : ({}) {} elements:".format(msg, colltype.__name__, len(data)))
            for count, (name, datum) in enumerate(zip(names, data)):
                if is_collection(datum):
                    data_summary(datum, data_index=name, recursion_depth=recursion_depth)
                else:
                    single_data_summary(datum, data_index=name, recursion_depth=recursion_depth)
                if count == coll_len_lim:
                    debug("...")
                    break
    else:
        # illustrate the data
        if msg:
            debug("{} :".format(msg))
        single_data_summary(data, data_index, recursion_depth=recursion_depth)

def shapes_list(thelist):
    try:
        return [get_shape(x) for x in thelist]
    except:
        # non-ndarray case
        return [len(x) for x in thelist]

def get_shape(element):
    """numpy / scipy csr shape fetcher, handling empty inputs"""
    return element.shape if element.size > 0 else ()

def lens_list(thelist):
    return [len(x) for x in thelist]


def read_lines(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line.strip())
    return lines


# list flattener
def flatten(llist):
    return [value for sublit in llist for value in sublit]


# converts elements of l to the index they appear in the reference
# flattens, if necessary
def align_index(input_list, reference):
    output = []
    for l in input_list:
        try:
            iter(l)
            # iterable, recurse
            res = align_index(l, reference)
            output.append(res)
        except:
            output.append(reference.index(l))
    # return np.array_split(np.array(output), len(output))
    return output

def count_label_occurences(labels, return_only_majority=False):
    """Gets majority (in terms of frequency) label in (potentially multilabel) input
    """
    counts = Counter()
    try:
        # iterable labels
        labels[0].__iter__
        for lab in labels:
            counts.update(lab)
    except AttributeError:
        # non-iterable labels
        counts = Counter(labels)
    # just the max label
    if return_only_majority:
        return counts.most_common(1)[0][0]
    return sorted(list(counts.most_common()), key= lambda x: x[1], reverse=True)


# split lists into sublists
def sublist(llist, sublist_length, only_index=False):
    if sublist_length == 1:
        if only_index:
            return [list(range(len(llist)))]
        return [llist]
    divisions = range(0, len(llist), sublist_length)
    if only_index:
        # just the lengths
        return [len(d) for d in divisions]
    return [llist[i:i + sublist_length] for i in divisions]


# print elapsed time to string
def elapsed_str(previous_tic, up_to=None):
    if up_to is None:
        up_to = time.time()
    duration_sec = up_to - previous_tic
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# datetime for timestamps
def datetime_str():
    return time.strftime("%d%m%y_%H%M%S")


# Logging
def need(condition, msg):
    if not condition:
        error(msg)


def error(msg, condition=True):
    if not condition:
        return
    logger = logging.getLogger()
    logger.error(msg)
    raise Exception(msg)


def info(msg):
    logger = logging.getLogger()
    logger.info(msg)


def debug(msg):
    logger = logging.getLogger()
    logger.debug(msg)


def debug2(msg):
    logger = logging.getLogger()
    logger.debug(msg)


def warning(msg):
    logger = logging.getLogger()
    logger.warning("[!] " + msg)

# read pickled data
def read_pickled(path, defaultNone=False):
    """Pickle deserializer function
    """
    if defaultNone:
        if not exists(path):
            return None
    info("Reading serialized from {}".format(path))
    with open(path, "rb") as f:
        return pickle.load(f)

# write pickled data
def write_pickled(path, data):
    """Pickle serializer function
    """
    info("Serializing to {}".format(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)

# yaml ordered read / write

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

# object to store times for tic-tocs
class tictoc:
    """Compound statement class to time a block of code
    """
    start = None
    func = None
    msg = None
    do_print = True
    announce = True
    history = []

    def __init__(self, msg, printer_func=logging.getLogger().info, do_print=True, announce=True):
        self.msg = msg
        self.func = printer_func
        self.do_print = do_print
        self.announce = announce

    def __enter__(self):
        self.start = time.time()
        if self.announce:
            self.func(">>> Starting: {}".format(self.msg))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # convert to smhd
        elapsed = elapsed_str(self.start)
        msg = "<<< {} took {}.".format(self.msg, elapsed)
        self.func(msg)
        self.history.append(msg)

    @staticmethod
    def log(outfile):
        """Writing all recorded times to a file
        """
        with open(outfile, "w") as f:
            f.write("\n".join(tictoc.history))
        info("Timings logged in {}".format(outfile))

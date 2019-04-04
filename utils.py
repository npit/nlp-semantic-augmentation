import time
import pickle
import logging
import numpy as np
import os
from collections import namedtuple
import nltk

num_warnings = 0


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
        conf = namedtuple(ntname, keys)(*[to_namedtuple(conf_dict[k], "dummy") if type(conf_dict[k]) == dict else conf_dict[k] for k in keys])
    else:
        conf = namedtuple(ntname, keys)(*[conf_dict[k] for k in keys])
    return conf


def as_list(x):
    """Convert the input to a single-element list, if it's not a list
    """
    return [x] if type(x) != list else x


# function for one-hot encoding, can handle multilabel
def one_hot(labels, num_labels):
    output = np.empty((0, num_labels), np.float32)
    for annot in labels:
        binarized = np.zeros((1, num_labels), np.float32)
        if type(annot) == list:
            for lbl in annot:
                binarized[0, lbl] = 1.0
        else:
            binarized[0, annot] = 1.0
        output = np.append(output, binarized, axis=0)
    return output


def shapes_list(thelist):
    return [x.shape for x in thelist]


def lens_list(thelist):
    return [len(x) for x in thelist]


def read_lines(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line.strip())
    return lines


# converts elements of l to the index they appear in the reference
# flattens, if necessary
def align_index(input_list, reference):
    output = []
    for l in input_list:
        if type(l) == list:
            res = align_index(l, reference)
            output.append(res)
        else:
            output.append(reference.index(l))
    return np.array_split(np.array(output), len(output))


def get_majority_label(inp, num_labels, is_multilabel):
    """Gets majority (in terms of frequency) label in (potentially multilabel) input
    """
    maxfreq, maxlabel = -1, -1
    for t in range(num_labels):
        if is_multilabel:
            freq = len([1 for x in inp if t in x])
        else:
            freq = len([1 for x in inp if x == t])
        if freq > maxfreq:
            maxfreq = freq
            maxlabel = t
    return maxlabel


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
    logger.warning(msg)


# read pickled data
def read_pickled(path):
    """Pickle deserializer function
    """
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

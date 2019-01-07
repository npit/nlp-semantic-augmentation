import time
import pickle
import logging
import numpy as np


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


# converts elements of l to the index they appear in the reference
# flattens, if necessary
def align_index(input, reference):
    output = []
    for l in input:
        if type(l) == list:
            res = align_index(l, reference)
            output.append(res)
        else:
            output.append(reference.index(l))
    return output


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
    info("Reading serialized from {}".format(path))
    with open(path, "rb") as f:
        return pickle.load(f)


# write pickled data
def write_pickled(path, data):
    info("Serializing to {}".format(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)


# object to store times for tic-tocs
class tictoc:
    start = None
    func = None
    msg = None
    do_print = True
    announce = True

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
        self.func("<<< {} took {}.".format(self.msg, elapsed))

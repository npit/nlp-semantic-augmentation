import time
import pickle
import logging


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
    return [llist[i:i+sublist_length] for i in divisions]


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
def error(msg):
    logger = logging.getLogger()
    logger.error(msg)
    raise Exception(msg)
def info(msg):
    logger = logging.getLogger()
    logger.info(msg)
def debug(msg):
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
class Timer:
    times = []


# matlabeqsque timing function start
def tic():
    Timer.times.append(time.time())


# timing function end
def toc(msg):
    # convert to smhd
    elapsed = elapsed_str(Timer.times.pop())
    logger = logging.getLogger()
    logger.info("{} took {}.".format(msg, elapsed))

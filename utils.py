import time
import logging

# print elapsed time to string
def elapsed_str(previous_tic, up_to = None):
    if up_to is None:
        up_to = time.time()
    duration_sec = up_to - previous_tic
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# datetime for timestamps
def datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")

# error and quit quick function
def error(msg):
    logger = logging.getLogger()
    logger.error(msg)
    raise Exception(msg)

# object to store times for tic-tocs
class Timer:
    times = []

# matlabeqsque timing function start
def tic():
    Timer.times.append(time.time())

# timing function end
def toc(msg):
    logger = logging.getLogger()
    # convert to smhd
    elapsed = elapsed_str(Timer.times.pop())
    logger.info("{} took {}.".format(msg, elapsed))

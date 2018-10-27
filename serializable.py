from utils import error, read_pickled, write_pickled, info, debug
from os.path import exists, isfile
from os import makedirs


class Serializable:
    name = None
    serialization_dir = None
    loaded_raw = False
    loaded_raw_preprocessed = False
    loaded_preprocessed = False

    def __init__(self, serialization_dir):
        self.serialization_dir = serialization_dir

    # set paths according to serializable name
    def set_paths(self, name = None):
        if name is None:
            name = self.name
        if not exists(self.serialization_dir):
            makedirs(self.serialization_dir, exist_ok=True)
        # raw
        self.serialization_path = "{}/raw_{}.pickle".format(self.serialization_dir, name)
        # preprocessed
        self.serialization_path_preprocessed = "{}/{}.preprocessed.pickle".format(self.serialization_dir, name)

    def acquire(self, fatal_error = True, do_preprocess=True):
        self.set_paths(self.name)
        # if fitting preprocessed data already exists, load them and we're done
        debug("Checking preprocessed path:{}".format(self.serialization_path_preprocessed))
        if exists(self.serialization_path_preprocessed) and isfile(self.serialization_path_preprocessed):
            preprocessed = read_pickled(self.serialization_path_preprocessed)
            self.handle_preprocessed(preprocessed)
            self.loaded_preprocessed = True
            return True

        # no preprocessed data exists. Check if serialized data exists
        if exists(self.serialization_path) and isfile(self.serialization_path):
            debug("Checking serialized path:{}".format(self.serialization_path))
            self.loaded_raw_serialized = True
            raw_serialized = read_pickled(self.serialization_path)
            self.handle_raw_serialized(raw_serialized)
        else:
            # nothing serialized exists - fetch the raw dataset
            raw = self.fetch_raw()
            if raw is None:
                if fatal_error:
                    error("Unable to load serializable {}".format(self.name))
                else:
                    return False
            # serialize it the raw data
            self.loaded_raw = True
            write_pickled(self.serialization_path, raw)
            self.handle_raw(raw)

        if do_preprocess:
            # process the acquired data
            self.preprocess()
            #info("Done loading serializable {}".format(self.name))
        return True

    def handle_preprocessed(self, preprocessed):
        error("Need to override preprocesseraw path dataset setter for {}".format(self.name))

    def set_raw_path(self):
        error("Need to override raw path dataset setter for {}".format(self.name))

    def fetch_raw(self):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def load_serialized(self):
        error("Need to override serialized data loader for {}".format(self.name))

    def handle_raw_serialized(self, raw_serialized):
        error("Need to override raw serialized data handler for {}".format(self.name))

    def preprocess(self):
        error("Need to override preprocessing function for {}".format(self.name))

    def get_all_raw(self):
        error("Need to override raw data getter for {}".format(self.name))

    def get_all_preprocessed(self):
        error("Need to override preprocessed data getter for {}".format(self.name))


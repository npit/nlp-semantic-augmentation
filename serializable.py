from utils import error, read_pickled, write_pickled, info, debug
from os.path import exists, isfile
from os import makedirs


class Serializable:
    name = None
    serialization_dir = None
    loaded_raw = False
    loaded_raw_preprocessed = False
    loaded_preprocessed = False
    # paths from where to load data, in priority order
    data_paths = []
    # corresponding read functions to acquire the data
    read_functions = []
    # corresponding processing functions to call on loaded data
    handler_functions = []
    # load flags
    load_flags = []

    def __init__(self, serialization_dir):
        self.serialization_dir = serialization_dir

    def set_serialization_params(self):
        # setup paths
        self.data_paths = self.set_paths_by_name(self.name) + [self.get_raw_path()]
        # alias some paths
        self.serialization_path_preprocessed, self.serialization_path = self.data_paths[:2]
        self.read_functions = [read_pickled, read_pickled, self.fetch_raw]
        self.handler_functions = [self.handle_preprocessed, self.handle_raw_serialized, self.handle_raw]

    # set paths according to serializable name
    def set_paths_by_name(self, name = None):
        if name is None:
            name = self.name
        if not exists(self.serialization_dir):
            makedirs(self.serialization_dir, exist_ok=True)
        # raw
        serialization_path = "{}/raw_{}.pickle".format(self.serialization_dir, name)
        # preprocessed
        serialization_path_preprocessed = "{}/{}.preprocessed.pickle".format(self.serialization_dir, name)
        return [serialization_path_preprocessed, serialization_path]

    def attempt_load(self, index):
        if self.data_paths[index] is None or (exists(self.data_paths[index]) and isfile(self.data_paths[index])):
            data = self.read_functions[index](self.data_paths[index])
            if data is None:
                debug("Failed to load {} from path {}".format(self.name, self.data_paths[index]))
                return False
            self.handler_functions[index](data)
            self.load_flags[index] = True
            return True
        else:
            debug("Failed to load {} from path {}".format(self.name, self.data_paths[index]))
            return False

    def acquire2(self, fatal_error = True, do_preprocess=True):
        self.load_flags = [False for _ in self.data_paths]
        for index in range(len(self.data_paths)):
            if (self.attempt_load(index)):
                return index
        if fatal_error:
            error("Failed to load {}".format(self.name))
        else:
            return False
        if do_preprocess:
            self.preprocess()
        return True


    def get_raw_path(self):
        error("Need to override raw path getter for {}".format(self.name))

    def handle_preprocessed(self, preprocessed):
        error("Need to override preprocesseraw path dataset setter for {}".format(self.name))

    def set_raw_path(self):
        error("Need to override raw path dataset setter for {}".format(self.name))

    def fetch_raw(self):
        error("Need to override raw data fetcher for {}".format(self.name))

    def handle_raw(self, raw_data):
        error("Need to override raw data handler for {}".format(self.name))

    def handle_raw_serialized(self, raw_serialized):
        error("Need to override raw serialized data handler for {}".format(self.name))

    def preprocess(self):
        error("Need to override preprocessing function for {}".format(self.name))

    def get_all_raw(self):
        error("Need to override raw data getter for {}".format(self.name))

    def get_all_preprocessed(self):
        error("Need to override preprocessed data getter for {}".format(self.name))


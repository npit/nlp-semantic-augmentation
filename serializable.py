from utils import error, read_pickled, write_pickled, info, debug
from os.path import exists, isfile, join
from os import makedirs

"""
Class to represent an object loadable from disk or computable,
used in the classification pipeline. It can have three successive states:
- raw: a format that requires specific object-dependent loading
- serialized: a serialized format with pickle
- preprocessed: a serialized format with pickle, directly usable in the next pipeline phase

raw formats should yield serialized object versions,
and applying preprocessing on the serialized object
should produce an object directly usable to the next
phase of the pipeline.

The raw, serialized and preprocessed object names (i.e. loadable file names)
are computed automatically from a base object name.

"""


class Serializable:
    name = None
    base_name = None
    serialization_dir = None
    loaded_raw = False

    # flags for data loading
    loaded_aggregated = False
    loaded_enriched = False
    loaded_finalized = False
    loaded_raw_serialized = False
    loaded_preprocessed = False

    # variables for serialization paths
    serialization_path_aggregated = None
    serialization_path_preprocessed = None
    serialization_path = None

    # paths from where to load data, in priority order
    data_paths = []
    # corresponding read functions to acquire the data
    read_functions = []
    # corresponding processing functions to call on loaded data
    handler_functions = []
    # load flags
    load_flags = []

    # paths to load necessary resources required to compute data from scratch
    resource_paths = []
    # corresponding reader and hanlder functions
    resource_read_functions = []
    resource_handler_functions = []
    resource_always_load_flag = []

    def __init__(self, dir_name):
        self.serialization_dir = join(self.config.folders.serialization, dir_name)
        self.raw_data_dir = join(self.config.folders.raw_data, dir_name)
        self.loaded_raw_serialized = False
        self.loaded_serialized = False
        self.loaded_preprocessed = False
        self.loaded_aggregated = False

    def set_serialization_params(self):
        # setup paths
        self.set_paths_by_name(self.name, raw_path=self.get_raw_path())
        # alias some paths
        self.serialization_path_preprocessed, self.serialization_path = self.data_paths[:2]
        self.read_functions = [read_pickled, read_pickled, self.fetch_raw]
        self.handler_functions = [self.handle_preprocessed, self.handle_raw_serialized, self.handle_raw]

    # set paths according to serializable name
    def set_paths_by_name(self, name=None, raw_path=None):
        if name is None:
            name = self.name
        if not exists(self.serialization_dir):
            makedirs(self.serialization_dir, exist_ok=True)
        # raw
        self.serialization_path = "{}/raw_{}.pickle".format(self.serialization_dir, name)
        # preprocessed
        self.serialization_path_preprocessed = "{}/{}.preprocessed.pickle".format(self.serialization_dir, name)
        debug("Path setter returning paths wrt name: {}".format(name))
        self.data_paths = [self.serialization_path_preprocessed, self.serialization_path, raw_path]

    # attemp to load resource from specified paths
    def attempt_load(self, index):
        path = self.data_paths[index]
        # either path is None (resource is acquired without one) or it's a file that will be loaded
        if path is None or (exists(path) and isfile(path)):
            debug("Attempting load of {} with {}.".format(path, self.read_functions[index]))
            data = self.read_functions[index](path)
            if data is None:
                debug("Failed to load {} from path {}".format(self.name, path))
                return False
            self.handler_functions[index](data)
            self.load_flags[index] = True
            return True
        else:
            debug("Failed to load {} from path {}".format(self.name, path))
            return False

    def acquire_resources(self):
        # Check if there are any required resources to load
        if self.resource_paths:
            for r, res in enumerate(self.resource_paths):
                info("Loading required resource {}/{}: {}".format(r + 1, len(self.resource_paths), res))
                read_result = self.resource_read_functions[r](res)
                self.resource_handler_functions[r](read_result)

    def acquire_data(self, fatal_error=True, do_preprocess=True):
        self.load_flags = [False for _ in self.data_paths]
        for index in range(len(self.data_paths)):
            if (self.attempt_load(index)):
                return index
        # no data was found to load
        if fatal_error and not self.resource_paths:
            error("Failed to load {}".format(self.name))
        else:
            self.acquire_resources()
            return False
        if do_preprocess:
            self.preprocess()
        return True

    # configure resources to load
    def set_resources(self):
        self.resource_paths = []
        self.resource_read_functions = []
        self.resource_handler_functions = []
        self.resource_always_load_flag = []

    def get_raw_path(self):
        error("Need to override raw path getter for {}".format(self.name))

    def handle_preprocessed(self, preprocessed):
        error("Need to override preprocessed handling for {}".format(self.name))

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


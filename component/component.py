from bundle.bundle import DataPool
from bundle.bundle import Consumes, Produces

from utils import as_list, error, info, write_pickled


"""Abstract class representing a computation pipeline component
"""


class Component:
    config = None
    # IO data pool
    data_pool = None
    # variables to hold inputs & output types
    produces = None
    consumes = None

    model = None
    model_loaded = False

    component_name = None
    # required input  from other chains
    required_finished_chains = []

    suspension_handler = None
    def suspend(self):
        self.suspension_handler(self)

    def get_consumption(self, chain_name):
        self.consumes = as_list(self.consumes) if self.consumes is not None else []
        res = []
        for pr in self.consumes:
            try:
                dtype, usage = pr
            except (ValueError, TypeError):
                dtype, usage = pr, None
            res.append(Consumes(dtype, usage, self.get_name(), chain_name))
        return res
    def get_production(self, chain_name):
        self.produces = as_list(self.produces) if self.produces is not None else []
        res = []
        for pr in self.produces:
            try:
                dtype, usage = pr
            except (ValueError, TypeError):
                dtype, usage = pr, None
            res.append(Produces(dtype, usage, self.get_name(), chain_name))
        return res


    def get_component_name(self):
        return self.component_name

    def get_full_name(self):
        return "({}|{})".format(self.get_component_name(), self.get_name())

    def get_name(self):
        return self.name

    def get_required_finished_chains(self):
        return self.required_finished_chains

    def configure_name(self, name=None):
        # set configured name to the output bundle
        # if name is None:
        #     self.set_source_name(self.component_name)
        # else:
        #     self.set_source_name(name)
        pass

    def assign_data_pool(self, data_pool):
        self.data_pool = data_pool

    def get_outputs(self):
        """Outputs getter"""
        return self.outputs

    def get_model(self):
        return self.model

    def __str__(self):
        return self.get_full_name()

    def run(self):
        """Component runner function"""
        self.set_serialization_params()

        # try loading component outputs from disk
        if not (self.config.misc.allow_output_deserialization and self.load_outputs_from_disk()):
            # if not available, fetch component inputs
            self.get_component_inputs()

            # try to load component's model from disk, if not already
            self.attempt_load_model_from_disk(failure_is_fatal=False)
            if not self.model_loaded:
                # if not available, build it from inputs
                self.attempt_build_model()
                # save it to disk
                self.save_model()

            # use the model and inputs to produce outputs
            self.produce_outputs()
            # save them to disk
            self.save_outputs()
        # assign produced outputs to the data pool
        self.set_component_outputs()

    # abstracts / defaults
    def get_component_inputs(self):
        error("Attempted to get inputs via abstract function.")

    def set_component_outputs(self):
        error("Attempted to set outputs via abstract function.")

    def produce_outputs(self):
        error("Attempted to produce outputs via abstract function.")

    def save_model(self):
        """Undefined by default"""
        # error("Attempted to save model via abstract function.")
        pass

    def save_outputs(self):
        """Undefined by default"""
        pass

    def load_outputs_from_disk(self):
        """Load the component's output from disk"""
        return False

    def load_model_from_disk(self):
        """Load the component's model from disk"""
        return False

    def attempt_load_model_from_disk(self, force_reload=False, failure_is_fatal=False):
        """Wrapper for model loading"""
        if not self.config.misc.allow_model_deserialization:
            return False
        if self.model_loaded and not force_reload:
            # already loaded
            return True
        # else call the loading function
        self.model_loaded = self.load_model_from_disk()
        error(f"Failed to load prebuilt model for {self.get_full_name()} but it's required", not self.model_loaded and failure_is_fatal)
        if self.model_loaded:
            info(f"Loaded prebuilt model for {self.get_full_name()}")
        return self.model_loaded

    def attempt_build_model(self):
        """Wrapper for model building"""
        self.check_model_building_resources()
        self.build_model_from_inputs()

    def check_model_building_resources(self):
        """Check potentially required resources for model building"""
        pass

    def build_model_from_inputs(self):
        """Construct the component's model"""
        # error("Attempted to invoke abstract component build function.")
        pass

from utils import error, warning, info
from threading import Lock
import json

class Trigger:
    pipelines = None
    execution_lock = None
    """Class to cause a pipeline execution"""
    def __init__(self, trigger_name, config, is_blocking=False, requires_loaded_models=False):
        self.trigger_name = trigger_name
        self.config = config
        self.pipelines = []
        self.is_blocking = is_blocking
        self.execution_lock = Lock() 
        self.requires_loaded_models = requires_loaded_models

    def requires_model_loading(self):
        """Whether the trigger requires all pipeline models to be loadable"""
        return self.requires_loaded_models

    def link_pipeline(self, pipeline):
        """Assign pipeline objects"""
        self.pipelines.append(pipeline)
        self.data_pool = pipeline.data_pool

    def setup(self):
        """Prepare the trigger to be firable"""
        pass

    def parse_json_input(self, data):
        """Parse json input"""
        # read inputs
        # inputs = []
        try:
            data = json.loads(data)
            # inputs = data["texts"]
            # info(f"Got {len(inputs)} text instances:")
        # except KeyError:
        #     pass
        except (json.JSONDecodeError, TypeError) as je:
            # return "Cannot JSON decode"
            return None, f"Malformed inputs: {je}."
        return data, ""
        # if len(inputs) == 0:
        #     return None, "Malformed inputs."
        # # config
        # config = {}
        # try:
        #     config = data["config"]
        # except KeyError:
        #     pass
        # info(json.dumps(data))
        # return [data, config], ""

    
    def arm(self):
        """Prime the trigger to be able to fire"""
        error("Attempted to access abstract trigger arming function.")

    def fire(self, data=None):
        """Cause pipeline execution"""
        outputs = []
        with self.execution_lock:
            info(f"{self.trigger_name} is firing!")
            self.data_pool.clear_data()
            try:
                if data is not None:
                    self.package_data(data)
                info("Executing pipeline(s).")
                for pipeline in self.pipelines:
                    res = pipeline.run()
                    outputs.append(res)

                # squeeze
                if len(outputs) == 1:
                    outputs = outputs[0]
            except Exception as ex:
                self.handle_execution_exception(ex)
                outputs = {"ERROR": str(ex)}
            self.clean_up_data()

        return outputs

    def package_data(self, data):
        """Add received data to the data pool, to make available to the pipeline"""
        pass
    def clean_up_data(self):
        """Add received data to the data pool, to make available to the pipeline"""
        pass
    def handle_execution_exception(self, ex):
        """By default, print exception message"""
        if self.config.raise_errors:
            raise(ex)
        warning(str(ex))

class ImmediateExecution(Trigger):
    """Dummy trigger for direct pipeline execution"""
    def arm(self):
        """Fires immediately"""
        self.fire()

    def handle_execution_exception(self, ex):
        """For normal execution, exceptions are  fatal"""
        raise(ex)
from utils import error, warning, info
from threading import Lock

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
            except Exception as ex:
                warning(str(ex))
                outputs = {"ERROR": str(ex)}
                raise ex
            self.clean_up_data()

        return outputs

    def package_data(self, data):
        """Add received data to the data pool, to make available to the pipeline"""
        pass
    def clean_up_data(self):
        """Add received data to the data pool, to make available to the pipeline"""
        pass

class ImmediateExecution(Trigger):
    """Dummy trigger for direct pipeline execution"""
    def arm(self):
        """Fires immediately"""
        self.fire()

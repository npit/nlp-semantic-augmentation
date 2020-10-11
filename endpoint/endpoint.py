from component.component import Component
import json
from flask import Flask, request
from threading import Lock
from bundle.datausages import *
from bundle.datatypes import *
import numpy as np
import defs
from utils import info, datetime_str

from component.trigger import Trigger

class IOEndpoint(Trigger):
    name = "rest-io"
    data_buffer = None
    buffer_lock = None

    outgoing_data = None

    def __init__(self, trigger_name, config):
        self.config = config
        super().__init__(trigger_name, config, requires_loaded_models=True)

        try:
            self.url = self.config.url
            self.port = self.config.port
        except AttributeError:
            self.url = "localhost"
            self.port = 9999

        self.data_buffer = []
        self.buffer_lock = Lock() 
        self.app = Flask(config.name)

        @self.app.route("/test")
        def ingest():
            try:
                if request.data:
                    data = json.loads(request.data)
                else:
                    data = list(request.args.keys())
            except (json.JSONDecodeError, TypeError):
                # return "Cannot JSON decode"
                return {"status": 500, "message": "Malformed inputs."}
            self.insert_to_data_buffer(data)
            results = self.fire()
            return json.dumps(results, ensure_ascii=False)

        @self.app.route('/')
        def hello_world():
            return 'Hello World!'

    def insert_to_data_buffer(self, data):
        with self.buffer_lock:
            self.data_buffer.append(data)


    def arm(self):
        """Prime the trigger to be able to fire"""
        info(f"Deploying execution trigger: {self.name}")
        self.data_pool.mark_as_reference_data()
        self.app.run(host=self.config.url, port=self.config.port)

    def fire(self):
        # can continue if there's inputs for ingestion
        data = self.pop_from_data_buffer()
        # submit for execution
        results = super().fire(data)
        return results

    def pop_from_data_buffer(self):
        # fetch an element from the buffer
        with self.buffer_lock:
            data = self.data_buffer.pop(0)
        return data

    def package_data(self, data):
        """Add received data to the data pool, to make available to the pipeline.
        Runs within the execution lock
        """
        num = len(data)
        txt = Text(data)
        # all test
        ind = Indices(np.arange(num), np.ones((num,),np.int32), [defs.roles.test])
        dp = DataPack(txt, ind)
        dp.chain = self.name
        # set source dependent on timestamp to prevent deserializations
        dp.source = f"{self.name}_{datetime_str()}"
        self.data_pool.add_data(dp)

    def clean_up_data(self):
        """Restore to data contents prior to arming"""
        self.data_pool.fallback_to_reference_data()

class DynamicEndpoint:
    # see
    # https://stackoverflow.com/questions/29247992/register-new-module-as-a-rest-api-in-python-flask-dynamically
    pass
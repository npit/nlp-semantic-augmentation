from component.component import Component
import json
from flask import Flask, request
from threading import Lock
from bundle.datausages import Indices, DataPack
from bundle.datatypes import Text, Numeric, Dictionary
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

            if request.data:
                data, msg = self.parse_json_input(request.data)
                if data is None:
                    return {"status": 500, "message": msg}
            else:
                data = list(request.args.keys())

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
        self.app.run(host=self.url, port=self.port)

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
        for key, value in data.items():
            if key == Text.name:
                # package input text data
                txt = Text(value)
                # all test
                ind = Indices(np.arange(len(value)), [defs.roles.test])
                dp = DataPack(txt, ind)
            elif key in [Dictionary.name, "config"]:
                dat = Dictionary(value)
                dp = DataPack(dat)
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

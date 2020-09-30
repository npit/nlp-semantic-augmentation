import json

from flask import Flask, request

app = Flask(__name__)


@app.route("/predict")
def predict():
    data = request.data
    return json.dumps({"data": data})


@app.route('/')
def hello_world():
    return 'Hello World!'

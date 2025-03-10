"""Main module."""

from flask import Flask, request
import pandas as pd

from ml import perform_pipeline



app = Flask(__name__)

@app.route('/')
def hello_world():
    """Returns a 'Hello, World!' message."""
    return {
        "message": 'Hello, World!'
    }



@app.route("/predict", methods=["POST"])
def predict():
    """Returns a prediction."""
    request_json: dict[str, dict[str] | list] = request.json

    settings = request_json["settings"]
    df = pd.DataFrame(request_json["data"])

    return perform_pipeline(
        df,
        settings["component_count"],
        settings["cluster_count_start"],
        settings["cluster_count_end"]
    ).to_json()

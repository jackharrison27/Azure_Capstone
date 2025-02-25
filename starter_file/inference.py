from azureml.core import Workspace
import os
import json
import joblib
from azureml.core.model import Model
import pandas as pd

def init():
    global model
    model_path = Model.get_model_path('automl-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:

        data = json.loads(raw_data)
        df = pd.DataFrame(data, index=[0])
        
        y = model.predict(df)
        return json.dumps({"Results": [int(y[0]]}'})

    except Exception as e:
        error = str(e)
        return error
# Predicting Wine Quality Capstone Project

## Project Overview
In this project, we use Azure and Azure ML studio to create models to predict wine scores from 1(worst) to 10(best). We developed models two ways: one using AutoML and a second using a custom logistic regression model with various hyperparameters tuned using HyperDrive. 


We create train models using AutoML, then choose and deploy the best model. The model endpoints are then consumed via REST API calls.


## Preliminary Steps

Before loading the dataset and developing the models, we must do a few steps:

 - Sign on to Microsoft Azure and launch Azure Machine Learning Studio
 - Create the workspaces
 - Create compute clusters
 - Create experiments
 
## Data 

### Overview

This dataset is from the UCI Machine Learning Repository and can be found here: https://archive.ics.uci.edu/ml/datasets/wine+quality

This dataset contains ~1600 examples with 11 independent variables and 1 dependent variable. The data includes things like fixed acidity, density, pH, and alcohol levels and the output variable, quality, based on sensory data. 

### Data upload

The data was uploaded using the AzureML Datasets UI. It was uploaded from a csv found on the UCI Machine Learning Repository to the Workspace Blob Storage. In both notebooks, I loaded the data by referencing it by name in Blob Storage, as shown below: 
```python
dataset_name = 'wine_quality'
dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
```

The dataset was uploaded to AzureML Datasets and is seen here: 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/data.png?raw=true)

Below is quick view of the data:
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/data-quick-view.png?raw=true)


## AutoML

### AutoML Settings

The AutoML experiment was set to find a best classification model based on the primary metric of Accuracy. This was chosen because as we classify wine quality from 1-10, we want to maximize the proportion of true results among the total number of predictions. We chose to have a max of 3 concurrent runs and to time out after 20 minutes to save compute resources over time. 

### Voting Ensemble Model

The AutoML experiment was completed in 19m 4s and chose the Voting Ensemble to be the best model. A voting ensemble algorithm taking a majority vote from several algorithms in order to achieve beter results than a single model would. It had an accuracy of 69.86%. 

The Details widget shows information regarding all the runs, including their status, duration, and best metric. 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/run-widget.png?raw=true)

Below is a scatter plot of accuracy of all the AutoML runs.
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/widget-accuracy.png?raw=true)

This shows all the scores of all the different metrics the best model could be scored on. Note that even though this model had the best accuracy, it might have not been the best in other metrics!
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/details.png?raw=true)

This shows a completed experiment with the best model and it's run ID. 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/finished-automl.png?raw=true)

This is the best model. As stated, it is a voting ensemble with an accuracy of almost 70%!
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/voting-ensemble.png?raw=true)

After model deployment, a healthy enpoint was also confirmed via AzureML Endpoints
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/healthy-endpoint.png?raw=true)



## HyperDrive

The HyperDrive experiment was set to find the best metric of Accuracy. We chose to do logistic regression with hyperparameters of C, to help regularization strengths, and max iterations, to help conversion. C was randomly sampled from a uniform distribution between 0 and 1. Max iterations was randomly sampled from 100, 150, 200, 500, 750, and 1000. We also used a bandit early termination policy to begin after 2 runs, re-evaluate every 3 runs, and with a slack factor of 0.2. 

### Logistic Regression Model

The best model that was developed with HyperDrive had an Accuracy of 72.81%. The best parameter values were  C:'0.6548736752745553' and max_iter: '100'. 

The Details widget shows information regarding all the runs, including their status, duration, and best metric. 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/hd-run-widget.png?raw=true)

Below is a scatter plot of the HyperDrive runs:
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/healthy-endpoint.png?raw=true)

Below is a scatter plot of the C value and max iter that shows the accuracy shifts:
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/hd-c.png?raw=true)

This shows the best models accuracy and best parameters. 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/hd-best-values.png?raw=true)



## Model Deployment

We chose to deploy the best AutoML model. This model was chosen to be deployed because of its high accuracy and ease of use. It was deployed via an Azure Container Instance with Authorization enabled. The Container Instance was deployed changing the default settings of 0.1 CPU cores and 0.5 GB of memory to instead use 0.5 CPU cores and 1 GB of memory. 

From the Endpoints page, we can see an example of how to consume the endpoint. 
```python
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = {
    "Inputs": {
        "data":
        [
            {
                "fixed acidity": "7.0",
                "volatile acidity": "0.4",
                "citric acid": "0.09",
                "residual sugar": "2.1",
                "chlorides": "0.1",
                "free sulfur dioxide": "8.0",
                "total sulfur dioxide": "41.0",
                "density": "0.9934",
                "pH": "3.12",
                "sulphates": "0.60",
                "alcohol": "9.0"
            },
        ]
    },
    "GlobalParameters": {
        "method": "predict"
    }
}

body = str.encode(json.dumps(data))

url = 'http://a9cfd26d-4053-4ad3-90a0-1c34f7618283.southcentralus.azurecontainer.io/score'
api_key = '8A2L05Lc1uw5x8aNyeGo9FU2sEjOSb5M' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

This URL describes the Swagger URI: http://a9cfd26d-4053-4ad3-90a0-1c34f7618283.southcentralus.azurecontainer.io/swagger.json

The inference.py script shown below returns a response: 

```python

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
        return json.dumps({'classification': int(y[0])})

    except Exception as e:
        error = str(e)
        return error

```

## Future Improvements

There are a few ways we can improve the models in the future:
 - Inlcude more data
 - Balance the dataset by having similar numbers of all scores
 - Apply some sort of average to values that are missing instead of removing them
 - Allows more iterations to run for both AutoML and HyperDrive runs
 - Include more options for Hyperparameters in HyperDrive runs
 - Enable deep learning for model development

## Screen Recording
https://youtu.be/pDRtid6ywpk

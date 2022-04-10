# Predicting Wine Score Capstone Project

## Project Overview
In this project, we use Azure and Azure ML studio to create models to predict wine scores from 1(worst) to 10(best). We developed models two ways: one using AutoML and a second using a custom logistic regression model with various hyperparameters tuned using HyperDrive. 


We create train models using AutoML, then choose and deploy the best model. The model endpoints are then consumed via REST API calls.


## Architectural Diagram
![alt text](https://github.com/jackharrison27/Azure_Machine_Learning_Operations/blob/master/screenshots/architectural_diagram.png?raw=true)


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

The dataset was uploaded to AzureML Datasets and is seen here: 
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/data.png?raw=true)

Below is quick view of the data:
![alt text](https://github.com/jackharrison27/Azure_Capstone/blob/master/screenshots/data-quick-view.png?raw=true)


## AutoML

The AutoML experiment was set to find a best model based on the primary metric of Accuracy.

### Voting Ensemble

The AutoML experiemtn was completed in 19m 4s and chose the Voting Ensemble to be the best model. A voting ensemble algorithm taking a majority vote from several algorithms in order to achieve beter results than a single model would. It had an accuracy of 69.86%. 

The Details widget shows information regarding all the runs, including their status, duration, and best metric. 


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

This dataset is from 

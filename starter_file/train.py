from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Workspace


ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name='wine_quality')


def preprocess(data): 
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("quality")
    return x_df, y_df
    
x, y = preprocess(dataset)
# TODO: Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

run = Run.get_context()

def main():
  # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    
    args = parser.parse_args()
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('output', exists_ok=True)
    joblib.dump(model, filename='outputs/hyperdrive_model.pkl')
    
if __name__ == '__main__':
    main()

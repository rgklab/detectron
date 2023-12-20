import torch
import pandas as pd
import numpy as np

def load_data(path: str, bool_par: bool = False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)
    if (bool_par == True):
        df = df.drop('hospitalid', axis=1)
    # Extract features and labels
    features = df.drop(columns=['y_true']).values
    labels = df['y_true'].values

    # Print the head of the DataFrame
    print("DataFrame Head:")
    print(df.head())
    return np.array(features), np.array(labels)

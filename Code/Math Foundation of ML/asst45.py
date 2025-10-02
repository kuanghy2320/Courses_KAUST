import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

df = pd.read_csv('./data/iris.csv')

# extract only two classes 'Iris-setosa' and 'Iris-versicolor'. Drop 'Iris-virginica'
df = df[df['species'] != 'Iris-virginica']

# make the labels 1 and 0
df['species'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)

# generate X and y tensors, adding the ``1'' feature for the bias
N, D = df.shape
X = torch.tensor(df.iloc[:, 0:D-1].values, dtype=torch.float32)
X = torch.cat((torch.ones((N,1)), X), dim=1)
y = torch.tensor(df.iloc[:, D-1].values, dtype=torch.float32)


# YOUR CODE HERE



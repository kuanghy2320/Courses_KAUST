import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# you may want to write your kmeans routine separately (in a kmeans.py file) and import it here: 
# from kmeans import kmeans

# read the iris data
df = pd.read_csv('/home/kuangh/Desktop/Code/Math Foundation of ML/data/iris.csv')
feature_names = df.keys()
X = df.iloc[:, 0:4].to_numpy()

# 3d scatter plot of training vectors
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[:, 0], X[:,1], X[:,2])
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])


# YOUR CODE GOES HERE

# You should cluster the data to get an assignmemt of the training vectors to group IDs
# add an argument ``c = avec'' to scatter to color the groups differently. avec is the assignment
# that your clustering generated
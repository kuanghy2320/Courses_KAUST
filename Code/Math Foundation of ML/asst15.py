import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read MNIST training data
df = pd.read_csv('data/mnist_train.csv')
X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
y = df.iloc[:, 0].to_numpy()                # labels of images


# plot the first dozen images from the data set
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1, xticks=[], yticks=[])
    image = X[i, :].reshape((28,28))
    plt.imshow(image, cmap='gray')


# YOUR CODE GOES HERE


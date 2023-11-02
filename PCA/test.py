import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model import PCA

iris = datasets.load_iris()
X, y = iris.data, iris.target

model = PCA(2)
model.fit(X)
X_projected = model.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
            c = y, edgecolors='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Principle Component 1')
plt.xlabel('Principle Component 2')

plt.colorbar()
plt.show()


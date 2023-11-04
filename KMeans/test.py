import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import KMeans

X, y = datasets.make_blobs(n_samples=400, n_features=2, centers=4, shuffle=True, random_state=902)
print(X.shape)

clusters = np.unique(y)
print(clusters)

model = KMeans(K=len(clusters), m_iters=500, plot_steps=True)
y_pred = model.predict(X)
# print(y_pred)

model.plot_step()
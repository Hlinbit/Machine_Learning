import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import Perceptron

X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = Perceptron(0.01, 1000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

acc = np.sum(prediction == y_test) / len(y_test)

print('acc = ', acc)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test)

X0_1 = np.amin(X_test[:, 0])
X0_2 = np.amax(X_test[:, 0])

X1_1 = -(X0_1 * model.weight[0] + model.bias) / model.weight[1]
X1_2 = -(X0_2 * model.weight[0] + model.bias) / model.weight[1]

ax.plot([X0_1, X0_2], [X1_1, X1_2], 'k')

ymin = np.amin(X_test[:, 1])
ymax = np.amax(X_test[:, 1])
ax.set_ylim(ymin=ymin - 3, ymax=ymax + 3)

plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import SVM

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1, random_state=23212)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = SVM()
model.fit(X_train, y_train)
predict = model.predict(X_test)

print(model.weight, model.bias)

def visual_svm():
    
    def get_hyperplane_value(w, b, off, x):
        return -(x * w[0] - b - off) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c = y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    y0_1 = get_hyperplane_value(model.weight, model.bias, 0, x0_1)
    y0_2 = get_hyperplane_value(model.weight, model.bias, 0, x0_2)

    y1_1 = get_hyperplane_value(model.weight, model.bias, 1, x0_1)
    y1_2 = get_hyperplane_value(model.weight, model.bias, 1, x0_2)

    y2_1 = get_hyperplane_value(model.weight, model.bias, -1, x0_1)
    y2_2 = get_hyperplane_value(model.weight, model.bias, -1, x0_2)

    ax.plot([x0_1, x0_2], [y0_1, y0_2], 'k')
    ax.plot([x0_1, x0_2], [y1_1, y1_2], 'k')
    ax.plot([x0_1, x0_2], [y2_1, y2_2], 'k')

    x1_min = np.amin(np.amin(X_train[:, 1]))
    x1_max = np.amin(np.amax(X_train[:, 1]))
    ax.set_ylim(x1_min, x1_max)

    plt.show()

visual_svm()
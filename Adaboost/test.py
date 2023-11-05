import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import Adaboost

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = Adaboost(n_clf=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy(y_test, y_pred)
print("Accuracy: ", acc)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import Logistic_Regression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = Logistic_Regression(0.0001, 100000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.sum(y_pred == y_test) / len(y_pred)

print(f'LR accurracy: {acc}')
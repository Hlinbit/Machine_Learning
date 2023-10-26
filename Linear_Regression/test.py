import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from model import LinearRegression 

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,8))
# plt.scatter(X[:, 0], y, color='b', marker='o', s = 20)
# plt.show()

model = LinearRegression(0.01, 10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def mse(y_true, y_predict):
    return np.mean((y_true - y_predict)**2)

mse_value = mse(y_test, y_pred)
print(mse_value)

y_pred_line = model.predict(X)
camp = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,8))
m1 = plt.scatter(X_train, y_train, c=camp(0.9), s=10)
m2 = plt.scatter(X_test, y_test, c=camp(0.5), s=10)

plt.plot(X, y_pred_line, color='black', linewidth=2, label='prediction')
plt.show()
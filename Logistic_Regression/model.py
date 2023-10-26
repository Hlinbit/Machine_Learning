import numpy as np

class Logistic_Regression:
    def __init__(self, lr=0.001, n_iter=5000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            linear_model = np.dot(X, self.weight) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Here we use cross entropy loss function
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        y_pred = self._sigmoid(y_pred)
        y_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 
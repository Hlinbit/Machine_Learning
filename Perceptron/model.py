import numpy as np

class Perceptron:
    def __init__(self, lr = 0.01, n_iters = 1000) -> None:
        self.weight = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_func


    def fit(self, X, y):
        n_samples, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0

        y_ = [1 if y_i > 0 else 0 for y_i in y] 

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight) + self.bias
                y_pred = self.activation_func(linear_output)

                update = self.lr * (y_[i] - y_pred)
                self.weight += update * x_i
                self.bias += update
                

    def predict(self, X):
        linear_output = np.dot(X, self.weight) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    def _unit_func(self, X):
        return np.where(X>=0, 1, 0)
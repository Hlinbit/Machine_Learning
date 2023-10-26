import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000) -> None:
        self.lr = lr
        self.n_iter = n_iters
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            pred_y = np.dot(X, self.weight) + self.bias

            dw = (1/n_samples) * np.dot(X.T, pred_y - y)
            db = (1/n_samples) * np.sum(pred_y - y)

            self.weight -= dw * self.lr
            self.bias -= db * self.lr
        
    def predict(self, x):
        y = np.dot(x, self.weight) + self.bias
        return y
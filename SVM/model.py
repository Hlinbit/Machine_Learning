import numpy as np

class SVM:
    def __init__(self, learning_rate = 0.001, colamada = 0.01, n_iters = 1000) -> None:
        self.lr = learning_rate
        self.lamada = colamada
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y<=0 , -1, 1)

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                dw, db = self._calculate(x_i, y_[idx])
                self.weight -= dw * self.lr
                self.bias -= db * self.lr

    def _calculate(self, x, y):
        linear_output = np.dot(x, self.weight) - self.bias
        condition = y * linear_output

        dw = 2 * self.lamada * self.weight - np.dot(x, y)
        db = y

        if condition >= 1:
            dw = 2 * self.lamada * self.weight
            db = 0

        return dw, db

    def predict(self, X):
        y = np.dot(X, self.weight) - self.bias
        return np.sign(y)

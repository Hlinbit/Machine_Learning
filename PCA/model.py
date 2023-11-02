import numpy as np

class PCA:
    def __init__(self, n_component) -> None:
        self.n_component = n_component
        self.component = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigen_value, eigen_vector = np.linalg.eig(cov)

        eigen_vector = eigen_vector.T
        order_idx = np.argsort(eigen_value)[::-1]
        eigen_value = eigen_value[order_idx]
        eigen_vector = eigen_vector[order_idx]

        self.component = eigen_vector[0:self.n_component]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.component.T)

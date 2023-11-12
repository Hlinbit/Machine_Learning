import numpy as np

class LDA:

    def __init__(self, n_component=2) -> None:
        self.n_component = n_component
        self.project_direction = None

    def fit(self, X, y): # X (150, 4)
        n_features = X.shape[1]
        labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            n_c = X.shape[0]
            S_B += (mean_diff).dot(mean_diff.T) * n_c

        A = np.linalg.inv(S_W).dot(S_B)
        e_value, e_vector = np.linalg.eig(A)
        e_vector = e_vector.T
        idxs = np.argsort(abs(e_value))[::-1]
        e_value = e_value[idxs]
        e_vector = e_vector[idxs]

        self.project_direction = e_vector[0: self.n_component]

    def transform(self, X):
        
        return np.dot(X, self.project_direction.T)
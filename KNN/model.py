import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt( np.sum( (x1 - x2)**2 ) )

class KNN:
    def __init__(self, k = 3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pred_label = [self._predict(x) for x in X]
        return np.array(pred_label)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_idx = np.argsort(distances)[0:self.k]
        k_nearest_label = [self.y_train[idx] for idx in k_idx]

        label = Counter(k_nearest_label).most_common(1)[0][0]
        return label
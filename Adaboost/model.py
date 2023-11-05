import numpy as np

class DecisionStump:
    def __init__(self) -> None:
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None 
        self.alpha = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        prediction = np.ones(n_samples)

        if self.polarity == 1:
            prediction[X_column < self.threshold] = -1
        else:
            prediction[X_column > self.threshold] = -1

        return prediction
    
class Adaboost:
     
    def __init__(self, n_clf=5) -> None:
        self.n_clf = n_clf
        self.clfs = []
    
    def fit(self, X ,y):
        n_samples, n_features = X.shape
        weight = np.full(n_samples, (1. / n_samples))

        for iter in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for thre in thresholds:
                    polarity = 1
                    prediction = np.ones(n_samples)
                    prediction[X_column < thre] = -1
                    
                    miss_weight = weight[prediction != y]
                    error = np.sum(miss_weight)

                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error
                        clf.polarity = polarity
                        clf.feature_idx = feature_i
                        clf.threshold = thre
            EPS = 1e-7
            clf.alpha = 0.5 * np.log((1 - error) / (error + EPS))
            
            prediction = clf.predict(X)
            weight *= (np.exp(-clf.alpha * prediction * y) / sum(weight))
            self.clfs.append(clf)
    
    def predict(self, X):
        prediction = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sum(prediction, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
import numpy as np

class NaiveBayes:
    
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        n_samples, n_feature = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # initial mean, var and priors
        self._mean = np.zeros((n_classes, n_feature),dtype=np.float64)
        self._var = np.zeros((n_classes, n_feature),dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for c in self._classes:
            X_c = X[(c == y), :]
            self._mean[c, :] = np.mean(X_c, axis=0)
            self._var[c, :] = np.var(X_c, axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
        print("mean = ", self._mean)
        print("var = ", self._var)
        print("priors = ", self._priors)
        print("self._classes = ", self._classes)
    
    def predict(self, X):
        return [self._predict(x) for x in X] 
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[c])
            class_condition = np.sum(np.log(self._pdf(x, c)))
            posterior = prior + class_condition
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
            
        
    def _pdf(self, x, c):
        mean = self._mean[c]
        var = self._var[c]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator =  np.sqrt(2 * np.pi * var)
        return numerator / denominator
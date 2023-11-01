import numpy as np
from collections import Counter

def entropy(y):
    div_count = np.bincount(y)
    ps = div_count / len(y)
    return np.sum([-p * np.log2(p) for p in ps if p > 0])

class Node:

    def __init__(self, threshold=None, feature=None, left=None, right=None, *, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def node_is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=10, nfeat=None) :
        self.min_sample_split=2
        self.max_depth = max_depth
        self.nfeat = nfeat
        self.root = None

    def fit(self, X, y):
        self.nfeat = X.shape[1] if not self.nfeat else min(self.nfeat, X.Shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_sample, n_feature = X.shape
        n_lable = len(np.unique(y))

        if (depth >= self.max_depth 
            or n_lable == 1
            or n_sample < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feature, self.nfeat, replace=False)
        best_threshold = 0.
        left_idx = []
        right_idx = []
        best_gain = -1
        best_feature = None

        for f_idx in feat_idxs:

            f_values = X[:, f_idx]
            threshold, gain = self._find_best_threshold(f_values, y)

            if best_gain < gain:
                best_threshold = threshold
                best_gain = gain
                best_feature = f_idx


        left_idx, right_idx = self._spilt_by_threshold(X[:, best_feature], best_threshold)

        depth += 1

        left = self._grow_tree(X[left_idx, :], y[left_idx], depth)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth)

        node = Node(threshold=best_threshold, feature=best_feature, left=left, right=right)

        return node


    def predict(self, X):
       return np.array([self._predict(self.root, x) for x in X])

    def _predict(self, node, x):
        
        if node.node_is_leaf():
            return node.value

        if x[node.feature] <= node.threshold :
            return self._predict(node.left, x)
        else:
            return self._predict(node.right, x)


    def _find_best_threshold(self, f_values, y):
        max_gain = 0.
        threshold = 0.
        uni_value = np.unique(f_values)

        for f_value in uni_value:
                left_idx, right_idx = self._spilt_by_threshold(f_values, f_value)
                ent_gain = self._calculate_entropy_gain(y, left_idx, right_idx)

                if ent_gain > max_gain:
                    max_gain = ent_gain
                    threshold = f_value
        
        return threshold, max_gain

    def _spilt_by_threshold(self, feature_values, threshold):
        left_idx = np.argwhere(feature_values <= threshold).flatten()
        right_idx = np.argwhere(feature_values > threshold).flatten()
        return left_idx, right_idx

    def _calculate_entropy_gain(self, y, left_idx, right_idx):
        left_y = y[left_idx]
        right_y = y[right_idx]

        left_ent = entropy(left_y) * (len(left_idx) / len(y))
        right_ent = entropy(right_y) * (len(right_idx) / len(y))
        ent = entropy(y)

        ent_gain = ent - left_ent - right_ent
        return ent_gain

    def _most_common_label(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label


import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    """
     Gini Impurity.
    """

    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        Time Complexity: O(n_features * n_samples^2) worst-case
        """
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predicting class labels for samples.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # --------------------------------------------------
    # Core Tree Logic
    # --------------------------------------------------

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or
                n_labels == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_indices = self._get_feature_indices(n_features)

        best_feature, best_thresh = self._best_split(X, y, feature_indices)

        if best_feature is None:
            return TreeNode(value=self._most_common_label(y))

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return TreeNode(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feature], thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, feature_column, threshold):
        parent_gini = self._gini(y)

        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold

        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idx]), len(y[right_idx])

        child_gini = (n_l / n) * self._gini(y[left_idx]) + \
                     (n_r / n) * self._gini(y[right_idx])

        return parent_gini - child_gini

    def _gini(self, y):
        """
        Gini Impurity:
        G = 1 - Σ p_i^2
        """
        proportions = [np.sum(y == c) / len(y) for c in np.unique(y)]
        return 1 - np.sum(np.square(proportions))

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _get_feature_indices(self, n_features):
        if self.max_features is None:
            return range(n_features)
        return np.random.choice(n_features, self.max_features, replace=False)

import pandas as pd
import numpy as np
import seaborn as sns
# https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion='gini', min_samples_split=2, impurity_stopping_threshold=1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None
        self.loss_function = self._entropy if criterion == 'entropy' else self._gini

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        print("Done fitting")

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _is_finished(self, depth, y):
        # TODO: add another stopping criteria for checking if it is homogenous enough already
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split
                or self._is_homogenous_enough(y)):
            return True
        # end TODO
        return False

    def _is_homogenous_enough(self, y):
        # check if y is homogenous enough by comparing the gini or entropy is small enough
        # TODO
        impurity = self.loss_function(y)
        if impurity <= self.impurity_stopping_threshold:
            return True
        result = False
        return result

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth, y):
            u, counts = np.unique(y, return_counts=True)
            most_common_Label = u[np.argmax(counts)]
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(
            self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(
            X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _gini(self, y):
        """
        Calculate the Gini impurity of a set of labels.

        Parameters:
        y (array-like): The set of labels.

        Returns:
        float: The Gini impurity value.
        """
        # Check if y is int or categorical
        if not np.issubdtype(y.dtype, np.integer):
            # If not integer, calculate unique categories and their counts
            unique_categorical, counts_integer = np.unique(
                y, return_counts=True)
            # Calculate proportions of each category
            proportions = counts_integer / len(y)
        else:
            # If y is integer, directly compute proportions using bincount
            proportions = np.bincount(y)/len(y)
        gini = 1 - np.sum([p ** 2 for p in proportions if p > 0])
        return gini

    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        y (array-like): The set of labels.

        Returns:
        float: The entropy value.
        """
        # Check if y is int or categorical
        if not np.issubdtype(y.dtype, np.integer):
            # If not integer, calculate unique categories and their counts
            unique_categorical, counts_integer = np.unique(
                y, return_counts=True)
            # Calculate proportions of each category
            proportions = counts_integer / len(y)
        else:
            # If y is integer, directly compute proportions using bincount
            proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        """
        Calculate the information gain of a split.

        Parameters:
        X (array-like): The feature values.
        y (array-like): The target values.
        thresh (float): The threshold value.

        Returns:
        float: The information gain value.
        """
        parent_loss = self.loss_function(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (
            n_left / n) * self.loss_function(y[left_idx]) + (n_right / n) * self.loss_function(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score': - 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):
    """
    Parameters:
    - n_estimators (int): The number of trees.
    - criterion (str, optional): The impurity criterion to use for tree splitting. Default is 'gini'.
    - min_samples_split (int, optional): The minimum number of samples required to split a node. Default is 2.
    - max_depth (int, optional): The maximum depth of the tree. Default is 10.
    - impurity_stopping_threshold (float, optional): The impurity threshold for stopping tree growth. Default is 1.

    Attributes:
    - n_estimators (int): The number of trees.
    - criterion (str): The impurity criterion used for tree splitting.
    - max_depth (int): The maximum depth of the tree.
    - min_samples_split (int): The minimum number of samples required to split a node.
    - impurity_stopping_threshold (float): The impurity threshold for stopping tree growth.
    - trees (list): List to store the decision trees of the random forest.
    """

    def __init__(self, n_estimators, criterion='gini', min_samples_split=2, max_depth=10, impurity_stopping_threshold=1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.trees = []

    def fit(self, X, y):
        """
    Fit the Random Forest model to the training data.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target labels.
    """

        for _ in range(self.n_estimators):
            # Subsample the data
            random_subspace = np.random.choice(
                len(X[0]), len(X[0]), replace=True)
            X_subspace = X[random_subspace]
            y_subspace = y[random_subspace]
            # Select random features
            n_features = int(np.sqrt(X.shape[1]))
            random_features = np.random.choice(
                X.shape[1], n_features, replace=False)
            # Fit a decision tree to the subspace
            tree = DecisionTreeModel(
                max_depth=self.max_depth, criterion=self.criterion, min_samples_split=self.min_samples_split, impurity_stopping_threshold=self.impurity_stopping_threshold)
            tree.fit(X_subspace[:, random_features], y_subspace)
            # Add the tree to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - array: The predicted class labels.
        """
        # Make predictions for each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Take the mode of the predictions
        most_common_elements = np.array(
            [Counter(column).most_common(1)[0][0] for column in predictions.T])
        return most_common_elements


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def confusion_matrix(y_test, y_pred):
    """
    Create a confusion matrix for the given test and predicted labels.

    Parameters:
    - y_test (array-like): The true labels.
    - y_pred (array-like): The predicted labels.

    Returns:
    - array: The confusion matrix.
    """
    # Get unique classes from labels
    labels = sorted(set(y_test))
    # Initialize confusion matrix
    matrix = [[0 for _ in labels] for _ in labels]
    # Map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    # Fill confusion matrix
    for true_label, pred_label in zip(y_test, y_pred):
        # Get indices of true and predicted labels
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]
        matrix[true_index][pred_index] += 1

    return matrix


def classification_report(y_test, y_pred):
    """
    Create a classification report for the given test and predicted labels.

    Parameters:
    - y_test (array-like): The true labels.
    - y_pred (array-like): The predicted labels.

    Returns:
    - str: The classification report.
    """
    # Get unique classes from labels
    labels = sorted(set(y_test))
    num_classes = len(labels)
    # We wills store our values in metrics dictionary
    metrics = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}

    for i in range(num_classes):
        TP = sum((y_test[j] == labels[i]) and (
            y_pred[j] == labels[i]) for j in range(len(y_test)))
        FN = sum((y_test[j] == labels[i]) and (
            y_pred[j] != labels[i]) for j in range(len(y_test)))
        FP = sum((y_test[j] != labels[i]) and (
            y_pred[j] == labels[i]) for j in range(len(y_test)))

        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) != 0 else 0
        support = sum((y_test[j] == labels[i])
                      for j in range(len(y_test)))

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1-score'].append(f1_score)
        metrics['support'].append(support)

    accuracy = accuracy_score(y_test, y_pred)

    # Construct the classification report text
    txt = "{:20} {:>12} {:>12} {:>12} {:>12} {:>12}\n".format(
        "", "precision", "recall", "f1-score", "support", "accuracy")
    for i in range(num_classes):
        txt += "{:20} {:12.2f} {:12.2f} {:12.2f} {:12} {:12.2f}\n".format(labels[i], metrics['precision'][i],
                                                                          metrics['recall'][i], metrics['f1-score'][i],
                                                                          metrics['support'][i], accuracy)

    return txt


def _test():

    df = pd.read_csv(
        'breast_cancer.csv')

    # Call the model with integer target variable
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gini Accracy: " + str(acc))

    print(classification_report(y_test, y_pred))

    clf = DecisionTreeModel(max_depth=10, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Entropy Accracy: " + str(acc))

    print(classification_report(y_test, y_pred))

    # call the model with categorical target variable
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gini Accracy: " + str(acc))

    print(classification_report(y_test, y_pred))

    # Run the RF model
    rfc = RandomForestModel(n_estimators=3)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("RF Model")
    print(classification_report(y_test, rfc_pred))
    print(accuracy_score(y_test, rfc_pred))


if __name__ == "__main__":
    _test()

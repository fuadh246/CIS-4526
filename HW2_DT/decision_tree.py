

import pandas as pd
import numpy as np
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets

from sklearn.model_selection import train_test_split

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

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        print("Done fitting")

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        # TODO: add another stopping criteria for checking if it is homogenous enough already
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self, y):
        # check if y is homogenous enough by comparing the gini or entropy is small enough
        # TODO
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth, y):
            u, counts = np.unique(y, return_counts = True)
            most_common_Label = u[np.argmax(counts)]
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y):
        #TODO
        # gini
        #end TODO
        return gini
    
    def _entropy(self, y):
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
    
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # end TODO
        return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        split = {'score':- 1, 'feat': None, 'thresh': None}

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

    def __init__(self, n_estimators, criterion='gini', min_samples_split=2, max_depth=10):
        # TODO: do NOT simply call the RandomForest model from sklearn
        self.criterion=criterion
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        # end TODO

    def fit(self, X, y):
        # TODO: call the underlying n_estimators trees fit method
        # end TODO


    def predict(self, X):
        # TODO:
        # call the predict method of the underlying trees, return the majority prediction
        return pred_y
        # end TODO

    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def confusion_matrix(y_test, y_pred):
    # TODO: return the number of TP, TN, FP, FN
    #
    result = np.array([[TP, FP], [FN, TN]])
    # end TODO
    return result
    

def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score and print them out
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]
    # TODO:

    #
    # end TODO
    return(txt)


def _test():

    df = pd.read_csv('breast_cancer.csv')
    
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

    print(classification_report(y_test,y_pred))

    clf = DecisionTreeModel(max_depth=10, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Entropy Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

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

    print(classification_report(y_test,y_pred))

    # Run the RF model
    rfc = RandomForestModel(n_estimators=3)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("RF Model")
    print(classification_report(y_test, rfc_pred))
    print(accuracy_score(y_test, rfc_pred))

if __name__ == "__main__":
    _test()

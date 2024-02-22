import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            if isinstance(self.lchild, DecisionNode):
                return self.lchild.predict(x_test)
            else:
                return self.lchild.prediction
        else:
            if isinstance(self.rchild, DecisionNode):
                return self.rchild.predict(x_test)
            else:
                return self.rchild.prediction

    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] < self.split:
            if isinstance(self.lchild, DecisionNode):
                return self.lchild.leaf(x_test)
            else:
                return self.lchild  # Return the leaf node
        else:
            if isinstance(self.rchild, DecisionNode):
                return self.rchild.leaf(x_test)
            else:
                return self.rchild  # Return the leaf node


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


def gini(y):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum( p**2 )

    
def find_best_split(X, y, loss, min_samples_leaf, max_features):
    n, p = X.shape
    best = {'feature': -1, 'split': -1, 'loss': loss(y)}

    feature_count = int(max_features * p)
    selected_features = np.random.choice(range(int(p)), size=feature_count, replace=False)
    
    for col in selected_features:
        # Randomly pick k indices from the feature column
        candidates = np.random.choice(n, 11)

        for split_idx in candidates:
            split_value = X[split_idx, col]
            yl = y[X[:, col] < split_value]
            yr = y[X[:, col] >= split_value]

            # Check if the minimum samples per leaf condition is met
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue

            # Calculate the weighted average of the loss function
            l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))

            if l == 0:
                return col, split_value
            if l < best['loss']:
                best = {'feature': col, 'split': split_value, 'loss': l}

    return best['feature'], best['split']
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, max_features=0.3, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = loss # loss function; either np.var for regression or gini for classification

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf or len(set(y)) == 1:
            return self.create_leaf(y)

        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf, self.max_features)

        if col == -1:
            return self.create_leaf(y)

        X_left, y_left = X[X[:, col] < split], y[X[:, col] < split]
        X_right, y_right = X[X[:, col] >= split], y[X[:, col] >= split]

        lchild = self.fit_(X_left, y_left)
        rchild = self.fit_(X_right, y_right)

        return DecisionNode(col=col, split=split, lchild=lchild, rchild=rchild)


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        predictions = []

        for record in X_test:
            pred = self.root.predict(record)
            predictions.append(pred)

        return predictions


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, max_features, loss=np.var)

    def score(self, X_test, y_test):
        """
        Return the R^2 of y_test vs predictions for each record in X_test
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, prediction=np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, max_features, loss=gini)

    def score(self, X_test, y_test):
        """
        Return the accuracy_score() of y_test vs predictions for each record in X_tes
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        mode_result = stats.mode(y, keepdims=True)
        y_mode = mode_result.mode[0]
        return LeafNode(y, prediction=y_mode)




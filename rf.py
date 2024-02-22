import numpy as np
from collections import defaultdict
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []


    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        n, _ = X.shape
        oob_indexes = []  # List to store OOB indexes for each tree

        for _ in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n, n, replace=True)
            oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices)
            oob_indexes.append(oob_indices)

            if isinstance(self, RandomForestRegressor621): # Regression
                tree = RegressionTree621(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)
            else:    # Classification
                self.n_classes = len(np.unique(y))
                tree = ClassifierTree621(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            # Fit the tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        # if self.oob_score:
        #     self.oob_score_ = self.compute_oob_score(X, y, oob_indexes)

    # def compute_oob_score(self, X, y, oob_indexes):
    #     """
    #     A helper function that compute the OOB validation score estimate and 
    #     store in self.oob_score_.
    #     """
    #     oob_predictions = []

    #     for i, tree in enumerate(self.trees):
    #         # Get leaf information for each OOB observation
    #         oob_leaf_info = tree.get_leaf_info(X[oob_indexes[i], :])
    #         oob_leaf_nodes = [info['leaf_node'] for info in oob_leaf_info]

    #         # Update OOB predictions for each observation
    #         for j, leaf_node in zip(oob_indexes[i], oob_leaf_nodes):
    #             oob_predictions[j] += leaf_node.prediction

    #     # Combine predictions from all trees for each OOB observation
    #     combined_oob_predictions = oob_predictions / len(self.trees)

    #     if self.is_regressor:
    #         # Calculate OOB R^2 score
    #         return r2_score(y[oob_indexes[0]], combined_oob_predictions[oob_indexes[0]])
    #     else:
    #         # Calculate OOB accuracy score
    #         return accuracy_score(y[oob_indexes[0]], combined_oob_predictions[oob_indexes[0]])

            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = self.trees
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = []
        
        for record in X_test:
            total_weights = 0
            total_num = 0
            for tree in self.trees:
                leaf_node = tree.root.leaf(record)
                total_num += leaf_node.n
                total_weights += leaf_node.n * leaf_node.prediction

            predictions.append(total_weights / total_num)

        return predictions

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        predictions = self.predict(X_test)
        return r2_score(y_test, predictions)
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = self.trees

    
    def predict(self, X_test) -> np.ndarray:

        predictions = []

        for record in X_test:
            class_counts = defaultdict(int)

            for tree in self.trees:
                leaf_node = tree.root.leaf(record)
                class_counts[leaf_node.prediction] += leaf_node.n

            max_count = 0
            max_class = 0
            for c, count in class_counts.items():
                if max_count < count:
                    max_count = count
                    max_class = c

            predictions.append(max_class)

        return np.array(predictions)

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

          
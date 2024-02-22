I implemented a basic version of random forests library, using code from the decision tree implementation from the previous library, where the accuracy is comparable to sklearn. In the file, I created both regression and classification random forests, as well as the out-of-bag (OOB) error.

In the random forest tree file, I built these two classes:

• RandomForestRegressor621: contains score and predict methods
• RandomForestClassifier621: contains score and predict methods.
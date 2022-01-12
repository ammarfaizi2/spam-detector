# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm

data = pd.read_csv("data.csv")
# print(data.info())

# Splitting our data into X and y.
X = data["v2"].values
y = data["v1"].values

# Splitting our data into training and testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
						    random_state=0)

# Converting String to Integer
cv = CountVectorizer() 
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Applying SVM algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train, y_train)

# Accuracy
print(classifier.score(X_test,y_test))

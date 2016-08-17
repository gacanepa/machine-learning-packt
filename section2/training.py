# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:29:30 2016

@author: amnesia
"""

from sklearn.datasets import load_iris
 # Import LinearSVC class
from sklearn.svm import LinearSVC
# Import KNeighborsClassifier class
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
iris = load_iris()

# Assign to variables for more convenient handling
X = iris.data
y = iris.target

# Create an instance of the LinearSVC classifier
clf = LinearSVC()
# Train the model
clf.fit(X, y)
# Get the accuracy score of the LinearSVC classifier
print clf.score(X, y)
# Predict the response given a new observation
print clf.predict([[ 6.3,  3.3,  6.0,  2.5]])

# Create an instance of KNeighborsClassifier
# The default number of K neighbors is 5.
# This can be changed by passing n_neighbors=k as argument
knnDefault = KNeighborsClassifier() # K = 5
# Train the model
knnDefault.fit(X, y)
# Get the accuracy score of KNeighborsClassifier with K = 5
print knnDefault.score(X, y)
# Predict the response given a new observation
print knnDefault.predict([[ 6.3,  3.3,  6.0,  2.5]])

# Let's try a different number of neighbors
knnBest = KNeighborsClassifier(n_neighbors=10) # K = 10
# Train the model
knnBest.fit(X, y)
# Get the accuracy score of KNeighborsClassifier with K = 10
print knnBest.score(X, y)
# Predict the response given a new observation
print knnBest.predict([[ 6.3,  3.3,  6.0,  2.5]])

# Let's try a different number of neighbors
knnWorst = KNeighborsClassifier(n_neighbors=100) # K = 100
# Train the model
knnWorst.fit(X, y)
# Get the accuracy score of KNeighborsClassifier with K = 100
print knnWorst.score(X, y)
# Predict the response given a new observation
print knnWorst.predict([[ 6.3,  3.3,  6.0,  2.5]])

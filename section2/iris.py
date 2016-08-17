# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:00:27 2016

@author: amnesia
"""

from sklearn.datasets import load_iris
iris = load_iris()

# The feature (column) names and the response
print iris.feature_names
print iris.target
print iris.target_names

# The object types of the feature matrix and the response array
print type(iris.data)
print type(iris.target)

# The shapes of samples and features
print iris.data.shape

# Uncomment to view scikit-learn and pandas versions
# print('The pandas version is {}.'.format(pandas.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

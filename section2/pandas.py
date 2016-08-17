# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:21:29 2016

@author: amnesia
"""
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

# Read file and attribute list into variable
data = pd.DataFrame(data=iris['data'],
                 	columns=iris['feature_names'])
print data.describe()

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:52:00 2017

@author: Afkar
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
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

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_X = LabelEncoder()
X[:, 0] = encoder_X.fit_transform(X[:, 0])
hotencoder = OneHotEncoder(categorical_features = [0])
X = hotencoder.fit_transform(X).toarray()
encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)

#Split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9873)
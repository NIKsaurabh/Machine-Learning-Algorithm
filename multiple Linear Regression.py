#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:57:22 2019

@author: saurabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
data=pd.read_csv('50_Startups.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values 
plt.scatter(data['Profit'].values,data['State'].values)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
X[:,3]=encoder.fit_transform(X[:,3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]

#splitting the dataset in training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set result
y_pred=regressor.predict(X_test)
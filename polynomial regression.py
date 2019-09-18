#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:00:29 2019

@author: saurabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data
data=pd.read_csv('Position_Salaries.csv')
#plt.scatter(data['Position'].values,data['Salary'].values)
#plt.xticks(rotation='vertical')

X=data.iloc[:,1:2].values
y=data.iloc[:,2].values

#fitting data to linear regression model
from sklearn.linear_model import LinearRegression
Lregressor=LinearRegression()
Lregressor.fit(X,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
Pregressor=PolynomialFeatures(degree=4)
X_poly=Pregressor.fit_transform(X)

Lregressor_2=LinearRegression()
Lregressor_2.fit(X_poly,y)

#visualising the linear regression results
plt.scatter(X,y,color='red')
plt.plot(X, Lregressor.predict(X))
plt.title("Truth or Bluff (Linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#visualising the polynomial regression results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, Lregressor_2.predict(Pregressor.fit_transform(X_grid)))
plt.title("Truth or Bluff (Polynomial regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#predicting new results using linear regression
print(Lregressor.predict(np.array([6.5]).reshape(1,-1)))

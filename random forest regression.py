#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:51:00 2019

@author: saurabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

#fitting random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X,y)

#predicting a new result
y_pred=regressor.predict(np.array([6.5]).reshape(1,-1))

#visualising the result
x_grid=np.arange(min(X),max(X),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title("truth or bluff (random forest regression)")
plt.xlabel("position lavel")
plt.ylabel("salary")
plt.show()

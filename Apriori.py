#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:00:31 2019

@author: saurabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transaction=[]
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
'''for i in range(0,7501):
    l=[]
    for j in range(0,20):
        l.append(str(dataset.values[i,j]))
    transaction.append(l)'''
#print(transaction)

#training apriori on the dataset
from apyori import apriori
rules=apriori(transaction, min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

result=list(rules)
listRules = [list(result[i][0]) for i in range(0,len(result))]
#print(result)
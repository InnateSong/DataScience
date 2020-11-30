# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:11:33 2020

@author: Deivydas Vladimirovas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Splitting the data into train and test sets (based on percentage)
msk = np.random.rand(len(df)) < 0.8 #random float (between 0 and 1) generated and if less than 0.8, it is True, else, its false
train = cdf[msk] #80% of data set is True and is taken if true only 
test = cdf[~msk] #opposite is happening, data where it was True in boolean array, are now false and the flase are true from before, thus ~20% data are True now

#TRAIN SET
#creating the instances and making the regression fit the lines
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) #multiple variobles used
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

#Test set
#testing variables
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) #predict instance
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


















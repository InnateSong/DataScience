# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:06:47 2020

@author: Deivydas Vladimirovas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv")

des = df.describe()
print(des)

#selecting features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

#visualising some of the data
cdf.hist()
plt.show()

#comparing some features to see if there are any relationships
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.ylabel("co2 emissions")
plt.xlabel("cylinders")
plt.show()

#Splitting the data into train and test sets (based on percentage)
msk = np.random.rand(len(df)) < 0.8 #random float (between 0 and 1) generated and if less than 0.8, it is True, else, its false
train = cdf[msk] #80% of data set is True and is taken if true only 
test = cdf[~msk] #opposite is happening, data where it was True in boolean array, are now false and the flase are true from before, thus ~20% data are True now

#ONLY showing the TRAINING portion of the data 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#MODELLING
from sklearn import linear_model
lm = linear_model.LinearRegression() #instance of linear regression generated
train_x = np.asanyarray(train[['ENGINESIZE']]) #converting the engine size data to an array
train_y = np.asanyarray(train[['CO2EMISSIONS']]) #converting the engine size data to an array
lm.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', lm.coef_)
print ('Intercept: ',lm.intercept_)

#plotting predicted line with regression
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
#lm.coef_[0][0] == literally just converting it to float (not array)
#lm.intercept_[0] == literally just converting it to float (not array)

#plot(X = x values, Y = coef*x-values + intercept)
#plot(x, y=mx+c)
plt.plot(train_x, lm.coef_[0][0]*train_x + lm.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#EVALUATION
from sklearn.metrics import r2_score
#test data 
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
#predicting using test data and see how accurate it is compared to actual values
test_y_hat = lm.predict(test_x) #predicted Y values
print(test_y_hat[:5], test_y[:5])
#%.2f means 2 decimal float after the the next % 
print("MAE: %.2f"% np.mean(np.absolute(test_y_hat - test_y)))#mean of the absolute values in the difference between predicted y values and actual y values of in bound training set
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2)) #mean squared 
print("R2-score: %.2f" % r2_score(test_y_hat , test_y)) #r*2 score from sklearn module 







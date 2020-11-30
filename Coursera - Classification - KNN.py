# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:08:42 2020

@author: Deivydas Vladimirovas
"""

"""K-NEAREST NEIGHBOUR classification method"""
#imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#laoding data
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv")
print(df.head())
#visualising data
print(df["custcat"].value_counts())
df.hist(column="income", bins=50)
plt.show()
plt.close()

#selecting data
print(df.columns)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
print(X[0:5])

y = df['custcat'].values
print(y[0:5])

#Normalising data
#transforming the data to standard form 
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#train test splitting 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#applying KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
#starting with k=4 for now
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

#predicting the test set
yhat = neigh.predict(X_test)
print(yhat[0:5])

#evaluating accuracy of k and looping to find best K value
from sklearn import metrics
print("Accuracy for k = 4:")
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat)) #comparison of the two sets of data 
print("Accuracy for k = 6:")
k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

#TESTING DIFFERENT NUMBER OF K's
Ks = 10
mean_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    print(n, "", metrics.accuracy_score(y_test, yhat))


plt.plot(range(1,Ks),mean_acc)
plt.legend('Accuracy ')
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1) 


#predicting using the highest score of accuracy from the k loop 
#starting with k=4 for now
k = mean_acc.argmax()+1
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

#predicting the test set
yhat = neigh.predict(X_test)
print(yhat[0:5])















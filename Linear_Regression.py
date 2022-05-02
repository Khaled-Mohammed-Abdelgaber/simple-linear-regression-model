# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:22:08 2022

@author: khali
"""


import pandas as pd 
import numpy as np
import sklearn 



#importing datasets 
datasets = pd.read_csv("Salary_Data.csv")
X = datasets.iloc[:,:-1].values
y = datasets.iloc[:,-1].values
y = y.reshape(-1,1)
#fill missing data with average value
from sklearn.impute import SimpleImputer
x_imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
X[:,:]= x_imputer.fit_transform(X[:,:])


#Scaling of features
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X[:,:] = sc_X.fit_transform(X[:,:])
# # sc_y = StandardScaler()
# y[:,:]= sc_y.fit_transform(y[:,:])

#spliting data to test and train datasets
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size= test_size,random_state= 0)


#creating linear regression model 
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train , y_train)
y_pred_test= regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

#visulalization and result ploting 
import matplotlib.pyplot as plt

# for trainign data
plt.scatter(X_train , y_train ,color = 'red')
plt.plot(X_train,y_pred_train,color = 'blue')
plt.title("Salary vs experience trainig")
plt.xlabel("years of experince")
plt.ylabel("Salary")
plt.show()


#for test
plt.scatter(X_test , y_test ,color = 'red')
plt.plot(X_test,y_pred_test,color = 'blue')
plt.title("Salary vs experience trainig")
plt.xlabel("years of experince")
plt.ylabel("Salary")
plt.show()




































 

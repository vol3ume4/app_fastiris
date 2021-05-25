# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:47:05 2021

@author: ssridhar
"""

# Importing necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
import numpy as np
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\GreatLearning\\PES ML-2\\Session 7\\fast_logreg')
os.getcwd()


# Importing the dataset
data = pd.read_csv('iris.csv')
data.head()
# Dictionary containing the mapping
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Encoding the target variables to integers
data = data.replace(['Setosa', 'Versicolor' , 'Virginica'],[0, 1, 2])

X = data.iloc[:, 0:-1] # Extracting the independent variables
y = data.iloc[:, -1] # Extracting the target/dependent variable
y.unique()

logreg = LogisticRegression() # Initializing the Logistic Regression model
logreg.fit(X, y) # Fitting the model

# save the model to disk
filename = 'logreg.pkl'
pickle.dump(logreg, open(filename, 'wb')) 

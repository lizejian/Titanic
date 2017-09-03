# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 18:54:43 2017

@author: lizejian
"""
#import module
from sklearn import linear_model
import pandas as pd
import numpy as np

#Load train data
input_train = pd.read_csv('traindata.cvs')
data_train = input_train.filter(regex = '|S*|P*|E*|A*|F*|').as_matrix()

#Build model
y = data_train[:, 0]
X = data_train[:, 1:]

model = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)

model.fit(X, y)

#Load test data
input_test = pd.read_csv('testdata.cvs')
data_test = input_test.filter(regex = 'SibSp|Pclass|Parch|Sex*|Emb*|Age*|Fare*').as_matrix()

#Fit
predictions = model.predict(data_test)
results = pd.DataFrame({'PassengerId': input_test['PassengerId'].as_matrix(),  
                        'Survived': predictions.astype(np.int32)})

#Save results
results.to_csv("logistic_regression_predictions.csv", index = False)

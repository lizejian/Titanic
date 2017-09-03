# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:09:33 2017

@author: lizejian
"""

#import module
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

#Train Data
#Load original data
train_original = pd.read_csv('data/train.csv')

#Convert categorical variable into dummy variables
Embarked_dummies = pd.get_dummies(train_original.Embarked, prefix = 'Embarked')
Sex_dummies = pd.get_dummies(train_original.Sex, prefix = 'Sex')

#Concatenate DateFrame
train_new = pd.concat([train_original, Embarked_dummies, Sex_dummies], axis = 1)

#RandomForest fit misssing data
df = train_new[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

known_age = df[df.Age.notnull()].as_matrix()
unknown_age = df[df.Age.isnull()].as_matrix()

X = known_age[:, 1:]
y = known_age[:, 0]

classifier = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
classifier.fit(X, y)

predictedAges = classifier.predict(unknown_age[:, 1:])

df.loc[(df.Age.isnull()), ('Age')] = predictedAges

train_new.Age = df.Age

#Normalization
scaler = preprocessing.StandardScaler()
age_scale = scaler.fit(train_new.Age)
train_new['Age_scaled'] = scaler.fit_transform(train_new.Age, age_scale)

fare_scale = scaler.fit(train_new.Fare)
train_new['Fare_scaled'] = scaler.fit_transform(train_new.Fare, fare_scale)

#Drop specific columns
train_new = train_new.drop(['PassengerId', 'Name', 'Age', 'Fare', 'Sex', 'Ticket', 'Cabin',
                               'Embarked'], axis = 1)

#Save DataFrame to cvs
train_new.to_csv('traindata.cvs', index = False)



##Test Data
#Load original data
test_original = pd.read_csv('data/test.csv')

#Convert categorical variable into dummy variables
Embarked_dummies = pd.get_dummies(test_original.Embarked, prefix = 'Embarked')
Sex_dummies = pd.get_dummies(test_original.Sex, prefix = 'Sex')

#Concatenate DateFrame
test_new = pd.concat([test_original, Embarked_dummies, Sex_dummies], axis = 1)

#RandomForest fit misssing data
df = test_new[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

unknown_age = df[df.Age.isnull()].as_matrix()

predictedAges = classifier.predict(unknown_age[:, 1:])

df.loc[(df.Age.isnull()), ('Age')] = predictedAges

test_new.Age = df.Age

test_new.loc[test_new.Fare.isnull(), 'Fare'] = 5.5

#Normalization
test_new['Age_scaled'] = scaler.fit_transform(test_new.Age, age_scale)
test_new['Fare_scaled'] = scaler.fit_transform(test_new.Fare, fare_scale)

#Drop specific columns
test_new = test_new.drop(['Name', 'Age', 'Fare', 'Sex', 'Ticket', 'Cabin',
                               'Embarked'], axis = 1)

#Save DataFrame to cvs
test_new.to_csv('testdata.cvs', index = False)

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:06:57 2017

@author: lizejian
"""

import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/train.csv')

#Ticket class
plt.figure(1) 
survived_Pclass = data_train.Pclass[data_train.Survived == 0].value_counts()
unsurvived_Pclass = data_train.Pclass[data_train.Survived == 1].value_counts()
df_Pclass = pd.DataFrame({'survived': survived_Pclass, 'unsurvived': unsurvived_Pclass})
df_Pclass.plot(kind = 'bar', stacked = False)
plt.title('Survived or not')
plt.xlabel('Ticket class')
plt.ylabel('Person numbers')
plt.show()

#Age
plt.figure(2)
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel('Age')
plt.grid(b = True, which = 'major', axis = 'y')
plt.title('Survived or not')
plt.show()

#Sex
plt.figure(3)
survived_Sex = data_train.Sex[data_train.Survived == 1].value_counts()
unsurvived_Sex = data_train.Sex[data_train.Survived == 0].value_counts()
df_Sex = pd.DataFrame({'survived': survived_Sex, 'unsurvived': unsurvived_Sex})
df_Sex.plot(kind = 'bar', stacked = True)
plt.title('Survived or not')
plt.xlabel('Sex')
plt.ylabel('Person numbers')
plt.show()

#SibSp
plt.figure(4)
survived_SibSp = data_train.SibSp[data_train.Survived == 1].value_counts()
unsurvived_SibSp = data_train.SibSp[data_train.Survived == 0].value_counts()
df_SibSp = pd.DataFrame({'survived': survived_SibSp, 'unsurvived': unsurvived_SibSp})
df_SibSp.plot(kind = 'bar', stacked = False)
plt.title('Survived or not')
plt.xlabel('Siblings / Spouses')
plt.ylabel('Person numbers')
plt.show()

#Parch
plt.figure(5)
survived_Parch = data_train.Parch[data_train.Survived == 1].value_counts()
unsurvived_Parch = data_train.Parch[data_train.Survived == 0].value_counts()
df_Parch = pd.DataFrame({'survived': survived_Parch, 'unsurvived': unsurvived_Parch})
df_Parch.plot(kind = 'bar', stacked = False)
plt.title('Survived or not')
plt.xlabel('Parch')
plt.ylabel('Person numbers')
plt.show()

#Fare
plt.figure(6)
data_train.Fare[data_train.Survived == 1].value_counts(sort = False).plot(kind = 'bar')
data_train.Fare[data_train.Survived == 0].value_counts(sort = False).plot(kind = 'bar')
plt.title('Survived or not')
plt.xlabel('Fare')
plt.ylabel('Ratio')
plt.legend(('survived', 'unsurvived'))
plt.show()

#Embarked
plt.figure(7)
survived_Embarked = data_train.Embarked[data_train.Survived == 1].value_counts()
unsurvived_Embarked = data_train.Embarked[data_train.Survived == 0].value_counts()
df = pd.DataFrame({'survived': survived_Embarked, 'unsurvived': unsurvived_Embarked})
df.plot(kind = 'bar', stacked = False)
plt.title('Survived or not')
plt.xlabel('Embarked')
plt.ylabel('Person numbers')
plt.show()
#Load lib
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')

#Load train and test dataset
train = pd.read_csv('E:/machine learing/Titanic/data/train.csv')
test = pd.read_csv('E:/machine learing/Titanic/data/test.csv')
PassengerId = test['PassengerId']

#Outlier detection by Tukey'method
detect_features = ['Age', 'SibSp', 'Parch', 'Fare']
k = 1.5
allOutlier = []
for feature in detect_features:
	Q1 = np.percentile(train[feature], 25)
	Q3 = np.percentile(train[feature], 75)
	IQR = Q3 - Q1
	upperBound = Q3 + k*IQR
	lowerBound = Q1 - k*IQR
	curOutlier = train[(train[feature] < lowerBound) | 
						(train[feature] > upperBound)].index
	allOutlier.extend(curOutlier)
outlier = [k for k, v in Counter(allOutlier).items() if v > 2]
train.drop(outlier, axis = 0).reset_index(drop = True)

#Check for null and missing values
train = train.fillna(np.nan)

#Feature analysis
train_len = len(train)
dataset = pd.concat(objs = [train, test], axis = 0).reset_index(drop = True)

#Fill Fare missing values with the median value
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

#Apply log to Fare to reduce skewness distribution
dataset['Fare'] = dataset['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Convert Sex into categorical value 0 for male and 1 for female
dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})

#Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
index_age = list(dataset['Age'][dataset['Age'].isnull()].index)
age_median = dataset['Age'].median()
for i in index_age :
    age_pred = dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) &
    (dataset['Parch'] == dataset.iloc[i]['Parch']) & 
    (dataset['Pclass'] == dataset.iloc[i]['Pclass']))].median()
    if not np.isnan(age_pred) :
        dataset.loc[i, 'Age'] = age_pred
    else :
        dataset.loc[i, 'Age'] = age_median

#Feature engineering
#Get Title from Name
dataset['Title'] = pd.Series([re.split('[\,\.]+', x)[1].strip() for x in dataset['Name']])

#Convert to categorical values Title 
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess',
	'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].map({'Master': 0, 'Miss': 1, 'Ms': 1 , 'Mme': 1, 
	'Mlle': 1, 'Mrs': 1, 'Mr': 2, 'Rare': 3})
dataset['Title'] = dataset['Title'].astype(int)

#Create a family size descriptor from SibSp and Parch
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Create new feature of family size
dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallFamily'] = dataset['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
dataset['LargeFamily'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

#Convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ['Title'])
dataset = pd.get_dummies(dataset, columns = ['Embarked'], prefix = 'Em')

#Create feature whether passengers had cabin
dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

#Ticket
dataset['Ticket'] = dataset['Ticket'].apply(lambda x: 'X' 
	if x.isdigit() else re.split(r'[\s]', x.replace('.', '').replace('/', ''))[0])

#Create categorical values for Pclass
dataset['Pclass'] = dataset['Pclass'].astype('category')
dataset = pd.get_dummies(dataset, columns = ['Pclass'], prefix = 'Pc')

#Drop useless variables
drop_features = ['PassengerId', 'Cabin', 'Name', 'Ticket']
dataset.drop(labels = drop_features, axis = 1, inplace = True)

#Separate train dataset and test dataset
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ['Survived'], axis = 1, inplace = True)

#Separate train features and label 
y_train = train['Survived'].astype(int)
X_train = train.drop(labels = ['Survived'], axis = 1)

#Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits = 10)

#Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state),
		random_state = random_state, learning_rate = 0.1))
classifiers.append(RandomForestClassifier(random_state = random_state))
classifiers.append(ExtraTreesClassifier(random_state = random_state))
classifiers.append(GradientBoostingClassifier(random_state = random_state))
classifiers.append(MLPClassifier(random_state = random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = [cross_val_score(classifier, X_train, y = y_train, 
    	scoring = 'accuracy', cv = kfold, n_jobs = 4) for classifier in classifiers]
cv_means = [cv_result.mean() for cv_result in cv_results]
cv_std = [cv_result.std() for cv_result in cv_results]

cv_res = pd.DataFrame({'CrossValMeans': cv_means, 'CrossValerrors': cv_std,
	'Algorithm': ['SVC', 'DecisionTree', 'AdaBoost', 'RandomForest', 'ExtraTrees',
	'GradientBoosting', 'MultipleLayerPerceptron', 'KNeighboors',
	'LogisticRegression', 'LinearDiscriminantAnalysis']})

g = sns.barplot('CrossValMeans', 'Algorithm', data = cv_res, 
	palette='Set3', orient = 'h', **{'xerr':cv_std})
g.set_xlabel('Mean Accuracy')
g = g.set_title('Cross validation scores')

#META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING
#Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {'base_estimator__criterion': ['gini', 'entropy'],
              'base_estimator__splitter': ['best', 'random'],
              'algorithm': ['SAMME','SAMME.R'],
              'n_estimators': [1,2],
              'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid = ada_param_grid, cv = kfold, 
	scoring = 'accuracy', n_jobs = 4, verbose = 1)

gsadaDTC.fit(X_train,y_train)

ada_best = gsadaDTC.best_estimator_

#Extra Trees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {'max_depth': [None],
              'max_features': [1, 3, 10],
              'min_samples_split': [2, 3, 10],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [False],
              'n_estimators': [100,300],
              'criterion': ['gini']}

gsExtC = GridSearchCV(ExtC, param_grid = ex_param_grid, cv = kfold, 
	scoring = 'accuracy', n_jobs = 4, verbose = 1)

gsExtC.fit(X_train, y_train)

ExtC_best = gsExtC.best_estimator_

# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {'max_depth': [None],
              'max_features': [1, 3, 10],
              'min_samples_split': [2, 3, 10],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [False],
              'n_estimators' :[100,300],
              'criterion': ['gini']}

gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, 
	scoring = 'accuracy', n_jobs = 4, verbose = 1)

gsRFC.fit(X_train, y_train)

RFC_best = gsRFC.best_estimator_

# Gradient boosting tunning
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss': ['deviance'],
              'n_estimators': [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC, param_grid = gb_param_grid, cv = kfold,
	scoring = 'accuracy', n_jobs = 4, verbose = 1)

gsGBC.fit(X_train, y_train)

GBC_best = gsGBC.best_estimator_

#SVC classifier
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100, 200, 300, 1000]}

gsSVMC = GridSearchCV(SVMC, param_grid = svc_param_grid, cv = kfold, 
	scoring = 'accuracy', n_jobs = 4, verbose = 1)

gsSVMC.fit(X_train, y_train)

SVMC_best = gsSVMC.best_estimator_

#Plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
	n_jobs = -1, train_size = np.linspace(.1, 1.0, 5)):
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel('Training examples')
	plt.ylabel('Score')
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, 
		y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
	plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', 
		label ='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', 
		label = 'Cross-validation score')
	plt.legend(loc = 'best')
	return plt

g = plot_learning_curve(gsRFC.best_estimator_, 'RF mearning curves', 
	X_train, y_train, cv = kfold)
g = plot_learning_curve(gsExtC.best_estimator_, 'ExtraTrees learning curves',
	X_train, y_train, cv = kfold)
g = plot_learning_curve(gsSVMC.best_estimator_, 'SVC learning curves',
	X_train, y_train, cv = kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_, 'AdaBoost learning curves',
	X_train, y_train, cv = kfold)
g = plot_learning_curve(gsGBC.best_estimator_, 'GradientBoosting learning curves',
	X_train, y_train, cv = kfold)

test_Survived_RFC = pd.Series(RFC_best.predict(test), name = 'RFC')
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name = 'ExtC')
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name = 'SVC')
test_Survived_AdaC = pd.Series(ada_best.predict(test), name = 'Ada')
test_Survived_GBC = pd.Series(GBC_best.predict(test), name = 'GBC')

#Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC, test_Survived_ExtC, 
	test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC], axis = 1)

g = sns.heatmap(ensemble_results.corr(), annot = True)

#Ensemble modeling
votingC = VotingClassifier(estimators = [('rfc', RFC_best), 
	('extc', ExtC_best), ('svc', SVMC_best), ('adac', ada_best),
	('gbc',GBC_best)], voting = 'soft', n_jobs = 4)

votingC = votingC.fit(X_train, y_train)

#Prediction
y_test = pd.Series(votingC.predict(test), name = 'Survived')

result = pd.concat([PassengerId, y_test], axis = 1)

result.to_csv('titanic_predictions.csv', index = False)
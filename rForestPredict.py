import numpy as np
import pandas as pd
import transform as t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

dataTrain = pd.read_csv("train.csv")
dataTest = pd.read_csv("test.csv")

dataTrain = t.transformations(dataTrain)
dataTest = t.transformations(dataTest)

X_all = dataTrain.drop(['Survived', 'PassengerId','Ticket','Embarked','Alone','Sex'], axis=1)
y_all = dataTrain['Survived']
X_all =  X_all.apply(lambda x: x/x.max(), axis=0)
X_all.to_csv('modifiedTrainSet.csv', index = False)

num_test = 0.05
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=11)

clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 10], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 20], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }


acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_ 
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

ids = dataTest['PassengerId']
X_test = dataTest.drop(['PassengerId','Alone','Sex','Embarked','Ticket'], axis=1)
X_test =  X_test.apply(lambda x: x/x.max(), axis=0)
predictions = clf.predict(X_test)


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)

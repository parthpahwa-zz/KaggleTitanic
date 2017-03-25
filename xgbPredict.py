import numpy as np
import pandas as pd
import transform as t
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split

dataTrain = pd.read_csv("train.csv")
dataTest = pd.read_csv("test.csv")

dataTrain = t.transformations(dataTrain)
dataTest = t.transformations(dataTest)

X_all = dataTrain.drop(['Survived', 'PassengerId','Embarked','Cabin','Alone','Sex','Ticket'], axis=1)
y_all = dataTrain['Survived']


X_all.to_csv('modifiedTrainSet.csv', index=False)
X_all =  X_all.apply(lambda x: x/x.max(), axis=0)
num_test = 0.05
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=11)

gbm = xgb.XGBClassifier(max_depth= 10, n_estimators=400, learning_rate=0.03).fit(X_train, y_train)
predictions = gbm.predict(X_test)
print(accuracy_score(y_test, predictions))

ids = dataTest['PassengerId']


X_test = dataTest.drop(['PassengerId','Alone','Sex','Embarked','Cabin','Ticket'], axis=1)
X_test =  X_test.apply(lambda x: x/x.max(), axis=0)
predictions = gbm.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictionsXGBoost.csv', index = False)

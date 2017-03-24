import pandas
import numpy as np

dataFrame = pandas.read_csv("train.csv")
dataTest = pandas.read_csv("test.csv")

print "TRAINING SET"
print dataFrame.columns.values
print dataFrame.describe()
print dataFrame.describe(include =['O'])

print dataFrame.info()
print dataFrame[["Sex","Pclass","Survived"]].groupby(["Sex","Pclass"]).mean()
print dataFrame[["Sex","Survived"]].groupby(["Sex"]).mean()

print dataFrame[["Pclass","Survived"]].groupby(["Pclass"]).mean()
print dataFrame[["Age","Sex","Survived"]].groupby( [pandas.cut(dataFrame["Age"], np.arange(0,100,10) ), "Sex"] ).mean()


print "\n\n\nTEST SET"
print dataTest.columns.values
print dataTest.describe()
print dataTest.describe(include =['O'])

print dataTest.info()

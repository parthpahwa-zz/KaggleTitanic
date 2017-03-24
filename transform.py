import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
from sklearn.feature_extraction import DictVectorizer

def sexEncoding(str):
	if(str == "male"):
		return 1,0
	return 0,1

def embarkedEncoding(str):
	if (str == 'S'):
		return 1,0,0
	elif (str == 'C'):
		return 0,1,0
	return 0,0,1

def ageEncoding(str):
	if(0<str<=10):
		return 1,0,0,0,0,0,0,0
	elif(10<str<=20):
		return 0,1,0,0,0,0,0,0
	elif(20<str<=30):
		return 0,0,1,0,0,0,0,0
	elif(30<str<=40):
		return 0,0,0,1,0,0,0,0
	elif(40<str<=50):
		return 0,0,0,0,1,0,0,0
	elif(50<str<=60):
		return 0,0,0,0,0,1,0,0
	elif(60<str<=70):
		return 0,0,0,0,0,0,1,0
	return 0,0,0,0,0,0,0,1

def transformSex(df):
	x = df.Sex.apply(lambda x: sexEncoding(x)) 
	df["M"]= x.map(lambda x: x[0])
	df["F"]= x.map(lambda x: x[1])
	return df

def transformEmbarked(df):
	x = df.Sex.apply(lambda x: embarkedEncoding(x)) 
	df["S"]= x.map(lambda x: x[0])
	df["C"]= x.map(lambda x: x[1])
	df["Q"]= x.map(lambda x: x[2])
	return df

def transformAge(df):
	x = df.Age.apply(lambda x: ageEncoding(x))
	df["0-10"] = x.map(lambda x: x[0])
	df["10-20"] = x.map(lambda x: x[1])
	df["20-30"] = x.map(lambda x: x[2])
	df["30-40"] = x.map(lambda x: x[3])
	df["40-50"] = x.map(lambda x: x[4])
	df["50-60"] = x.map(lambda x: x[5])
	df["60-70"] = x.map(lambda x: x[6])
	df["70-80"] = x.map(lambda x: x[7])
	return df

def transformations(df):
	df = transformSex(df)
	df = transformEmbarked(df)
	df = transformAge(df)
	return df


dataTrain = pd.read_csv("train.csv")
dataTest = pd.read_csv("test.csv")

dataTrain = transformations	(dataTrain)
dataTest = transformations	(dataTest)
print dataTrain.head()



import numpy as np
import pandas as pd


title = {
"Capt":       4,
"Col":        4,
"Major":      4,
"Jonkheer":   1,
"Don":        1,
"Sir" :       1,
"Dr":         0,
"Rev":        0,
"the Countess":1,
"Dona":       1,
"Mme":        2,
"Mlle":       2,
"Ms":         2,
"Mr" :        3,
"Mrs" :       2,
"Miss" :      2,
"Master" :    3,
"Lady" :      1
}

def transfromCabin(df):
	df.Cabin.fillna(0, inplace=True)
	
	df.loc[df.Cabin.str[0] == 'A', 'Cabin'] = 1
	df.loc[df.Cabin.str[0] == 'B', 'Cabin'] = 2
	df.loc[df.Cabin.str[0] == 'C', 'Cabin'] = 3
	df.loc[df.Cabin.str[0] == 'D', 'Cabin'] = 4
	df.loc[df.Cabin.str[0] == 'E', 'Cabin'] = 5
	df.loc[df.Cabin.str[0] == 'F', 'Cabin'] = 6
	df.loc[df.Cabin.str[0] == 'T', 'Cabin'] = 8
	df.loc[df.Cabin.str[0] == 'G', 'Cabin'] = 7
	
	return df

def transformName(df):
	df.Name = df.Name.map(lambda x:x.split(',')[1].split('.')[0].strip())
	df.Name = df.Name.map(title)
	return df

def sexEncoding(str):
	if(str == "male"):
		return 1,0
	return 0,1

def embarkedEncoding(str):
	if (str == 'S'):
		return 1
	elif (str == 'C'):
		return 2
	elif (str == 'Q'):
		return 3
	return 0

def ageEncoding(str):
	if(0<str<=18):
		return 0,0,0,0,1
	elif(18<str<=33):
		return 0,0,0,1,0
	elif(33<str<=50):
		return 0,0,1,0,0
	elif(50<str<=70):
		return 0,1,0,0,0
	return 1,0,0,0,0

def encodeTicket(str):
	str = str.split(' ')
	if( not str[-1].isdigit()):
		return 5
	if(int(str[-1])<100001):
		return 0
	elif (int(str[-1]) < 200001):
		return 1
	elif (int(str[-1]) < 300001):
		return 2
	return 4

def transformTicket(df):
	df.Ticket = df.Ticket.apply(lambda x: encodeTicket(x))
	return df

def transformSex(df):
	x = df.Sex.apply(lambda x: sexEncoding(x)) 
	df["M"]= x.map(lambda x: x[0])
	df["F"]= x.map(lambda x: x[1])
	return df

def transformEmbarked(df):
	df.Embarked = df.Embarked.apply(lambda x: embarkedEncoding(x)) 
	return df

def transformAge(df):
	df = findRightAge(df)
	x = df.Age.apply(lambda x: ageEncoding(x))
	df["0-18"] = x.map(lambda x: x[4])
	df["18-33"] = x.map(lambda x: x[3])
	df["33-50"] = x.map(lambda x: x[2])
	df["50-70"] = x.map(lambda x: x[1])
	df[">70"] = x.map(lambda x: x[0])
	return df

def famSize(df):
	df["FamSize"] = df.SibSp + df.Parch + 1
	df["Alone"] = 0 
	df.loc[df['FamSize'] == 1, 'Alone'] = 1
	return df

def prodClassAge(df):
	df["Class*Age"] = df.Pclass*df.Age
	return df

def transformFare (df):
	index = df['Fare'].index[df['Fare'].apply(np.isnan)]
	for indx in index:
		if(df["Pclass"].ix[indx] == 1 and df["Sex"].ix[indx] == "female") :
			df["Fare"].ix[indx] = 106.125798
		elif(df["Pclass"].ix[indx] == 2 and df["Sex"].ix[indx] == "female") :
			df["Fare"].ix[indx] = 21.970121
		elif(df["Pclass"].ix[indx] == 3 and df["Sex"].ix[indx] == "female") :
			df["Fare"].ix[indx] = 16.118810
		elif(df["Pclass"].ix[indx] == 1 and df["Sex"].ix[indx] == "male") :
			df["Fare"].ix[indx] = 67.226127
		elif(df["Pclass"].ix[indx] == 2 and df["Sex"].ix[indx] == "male") :
			df["Fare"].ix[indx] = 19.741782
		else :
			df["Fare"].ix[indx] = 12.661633
	return df

def findRightAge(df):
	index = df['Age'].index[df['Age'].apply(np.isnan)]
	for indx in index:
		if(df["Pclass"].ix[indx] == 1 and df["Sex"].ix[indx] == "female") :
			df["Age"].ix[indx] = 34.611765
		elif(df["Pclass"].ix[indx] == 2 and df["Sex"].ix[indx] == "female") :
			df["Age"].ix[indx] = 28.722973
		elif(df["Pclass"].ix[indx] == 3 and df["Sex"].ix[indx] == "female") :
			df["Age"].ix[indx] = 21.75
		elif(df["Pclass"].ix[indx] == 1 and df["Sex"].ix[indx] == "male") :
			df["Age"].ix[indx] = 41.281386
		elif(df["Pclass"].ix[indx] == 2 and df["Sex"].ix[indx] == "male") :
			df["Age"].ix[indx] = 30.740707
		else :
			df["Age"].ix[indx] = 26.507589
	return df 

def transformations(df):
	df = transformSex(df)
	df = transformEmbarked(df)
	df = transformAge(df)
	df = famSize(df)
	df = transformFare(df)
	df = prodClassAge(df)
	df = transformTicket(df)
	df = transformName(df)
	df = transfromCabin(df)
	return df

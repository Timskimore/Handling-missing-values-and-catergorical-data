
#importing the libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#importing dataset 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
#X = dataset.drop(['Purchased'],axis=1)
y = dataset.iloc[:,3].values

#handling missing data 
from sklearn.preprocessing import Imputer
#print(X.dtypes)
imputer = Imputer()
X[:,1:3]= imputer.fit_transform(X[:,1:3])

#Encoding catergorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


#Splitting the dataset 
from sklearn.model_selection import train_test_split
train_X, test_X,train_y, test_y = train_test_split(X,y, test_size = 0.2,random_state = 0) 

#feature scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)
#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
data=pd.read_csv('weatherAUS.csv')

#DATA PREPROCESSING
#TAKING CARE OF THE MISSING DATA
total=data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
miss_data=pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
data=data.drop((miss_data[miss_data['Percent']>0.1]).index,1)
data=data.drop(columns=['Date','Location','RISK_MM'])
data["Pressure9am"].fillna(method='ffill',inplace=True)
data["Pressure3pm"].fillna(method='ffill',inplace=True)
data["WindDir9am"].fillna(method='ffill',inplace=True)
data["WindGustDir"].fillna(method='ffill',inplace=True)
data["WindGustSpeed"].fillna(method='ffill',inplace=True)
data=data.dropna(axis=0,how='any')

X=data.iloc[:,0:16].values
Y=data.iloc[:,-1].values

#ENCODING CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x=LabelEncoder()
le_y=LabelEncoder()
onhe=OneHotEncoder(categorical_features=[3,5,6])
X[:,3]=le_x.fit_transform(X[:,3])
X[:,5]=le_x.fit_transform(X[:,5])
X[:,6]=le_x.fit_transform(X[:,6])
X[:,15]=le_x.fit_transform(X[:,15])
Y=le_y.fit_transform(Y)

X=onhe.fit_transform(X).toarray()

#SPLITTING THE DATSET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#APPLYING FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#IMPORTING LIBRARIES FOR ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#CREATING THE CLASSIFIER
classifier=Sequential()

#CREATING INPUT,OUTPUT AND HIDDEN LAYERS
classifier.add(Dense(output_dim=32,init='uniform',activation='relu',input_dim=61))

classifier.add(Dense(output_dim=32,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#FITTING THE CLASSIFIER
classifier.fit(X_train,Y_train,batch_size=20,nb_epoch=100)

#PREDICTING THE RESULTS
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#CREATING CONFUSION MATRIX TO COMPARE AND CHECK THE ACCURACY
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
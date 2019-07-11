import numpy as np
import pandas as pd

data=pd.read_csv('Churn_Modelling.csv') 
X=data.iloc[:,3:13].values
y=data.iloc[:,13].values

miss_data=data.isnull().sum().sort_values(ascending=False)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x=LabelEncoder()
X[:,1]=le_x.fit_transform(X[:,1])
X[:,2]=le_x.fit_transform(X[:,2])
onhe=OneHotEncoder(categorical_features=[1])
X=onhe.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(input_dim=12,init='uniform',activation='relu',output_dim=6))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x=X_train,y=Y_train,epochs=100,batch_size=10)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
train['Embarked']=train['Embarked'].fillna(value='U')
test=pd.read_csv('test.csv')
test['Embarked']=test['Embarked'].fillna(value='U')
X_train=train.iloc[:,[2,4,5,6,7,9,11]].values
y_train=train.iloc[:,1:2].values
X_test=test.iloc[:,[1,3,4,5,6,8,10]].values


from sklearn.preprocessing import Imputer
md=Imputer()
X_train[:,[2]]=md.fit_transform(X_train[:,[2]])
X_test[:,[2]]=md.fit_transform(X_test[:,[2]])
X_test[:,[5]]=md.fit_transform(X_test[:,[5]])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x=LabelEncoder()
onhe_x=OneHotEncoder(categorical_features=[0,6])
X_train[:,1]=le_x.fit_transform(X_train[:,1])
X_train[:,6]=le_x.fit_transform(X_train[:,6])
X_train=onhe_x.fit_transform(X_train).toarray()
X_test[:,1]=le_x.fit_transform(X_test[:,1])
X_test[:,6]=le_x.fit_transform(X_test[:,6])
X_test=onhe_x.fit_transform(X_test).toarray()
X_train=X_train[:,1:]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.svm import SVC
classifier=SVC(C=10,kernel='rbf',gamma=0.03,random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_pred=np.reshape(y_pred,(418,1))
pass_id=test.iloc[:,0].values
pass_id=np.reshape(pass_id,(418,1))

y_result=np.append(arr=pass_id,values=y_pred,axis=1)


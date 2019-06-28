import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('weatherAUS.csv')
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

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,Y_train)

from sklearn.tree import DecisionTreeClassifier
classifier_DT=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier_DT.fit(X_train,Y_train)

from sklearn.ensemble import RandomForestClassifier
classifier_RF=RandomForestClassifier(n_estimators=5000,criterion='entropy',random_state=0)
classifier_RF.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

y_pred_DT=classifier_DT.predict(X_test)

y_pred_RF=classifier_RF.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

cm_DT=confusion_matrix(Y_test,y_pred_DT)

cm_Rf=confusion_matrix(Y_test,y_pred_RF)

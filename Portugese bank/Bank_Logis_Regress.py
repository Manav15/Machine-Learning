#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
data=pd.read_csv('bank-additional-full.csv',delimiter=';')

#checking for missing data
miss_data=data.isnull().sum().sort_values(ascending=False)

#creating the dependent and independent variable
X=data.iloc[:,[0,1,2,3,5,6,10,11,-3,-4,-5,-6]].values
Y=data.iloc[:, -1].values

#Encoding the Categorical Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x=LabelEncoder()
le_y=LabelEncoder()
onhe=OneHotEncoder(categorical_features=[1,2,3,4,5])
X[:,1]=le_x.fit_transform(X[:,1])
X[:,2]=le_x.fit_transform(X[:,2])
X[:,3]=le_x.fit_transform(X[:,3])
X[:,4]=le_x.fit_transform(X[:,4])
X[:,5]=le_x.fit_transform(X[:,5])
Y=le_y.fit_transform(Y)
X=onhe.fit_transform(X).toarray()

#Splitting the dataset into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Scaling the training and test match
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)

#Applying the logistic Regression Model 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(C=5,solver='newton-cg')
classifier.fit(X_train,Y_train)

#Predicting the Results
y_pred=classifier.predict(X_test)

#Comparing the results and finding the accuracy 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

#Applying K-FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)
accuracies.mean()
accuracies.std()

#Parameter Tuning using Grid Search
from sklearn.model_selection import GridSearchCV
parameters=[{'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
gd_ser=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
gd_ser=gd_ser.fit(X_train,Y_train)
best_para=gd_ser.best_params_
best_score=gd_ser.best_score_


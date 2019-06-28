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


#CREATING THE CLASSIFIER AND FITTING THE MODEL
from sklearn.ensemble import RandomForestClassifier
classifier_RF=RandomForestClassifier(n_estimators=5000,criterion='entropy',random_state=0)
classifier_RF.fit(X_train,Y_train)


#PREDICITING THE RESULT(WILL IT RAIN OR NOT)
y_pred_RF=classifier_RF.predict(X_test)

#CHECKING THE RESULT USING CONFUSION MATRIX FOR BINARY CLASSIFICATION
from sklearn.metrics import confusion_matrix
cm_Rf=confusion_matrix(Y_test,y_pred_RF)


"""**Due to computational limitations K-FOLD CROSS VALIDATION and 
GRID SEARCH WERE NOT ABLE TO RUN PROPERLY"""

#EVALUATING MODEL PERFORMANCE
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier_RF,X=X_train,y=Y_train,cv=10)
accuracies.mean()
accuracies.std()


#PARAMETER TUNING USING GRID SEARCH
from sklearn.model_selection import GridSearchCV
parameters=[{'n_estimators':[100,500,1000,2000,5000],'criterion':['gini','entropy']}]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,score='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,Y_train)
best_scores=grid_search.best_score_
best_paramters=grid_search.best_params_


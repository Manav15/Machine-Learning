#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values
y=y.reshape(-1,1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
 
#Fitting SVR to dataset
#creating regressor
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#Predicting the results
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#Visualizing the results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('TRUTH OR BLUFF(SVR)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
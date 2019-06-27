#polynomial regression

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2].values

'''#spliting dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)'''

#Fitting Linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

#visualizing Linear regression results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualising polynomial regression results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting new results by linear regression model
y_pred=lin_reg.predict(6.5)

#predicting new results by polynomial regression model
y_pred_2=lin_reg_2.predict(poly_reg.fit_transform(6.5))
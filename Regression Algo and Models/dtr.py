#Decision Tree Regression
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#creating the regressor
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting the results
y_pred=regressor.predict(6.5)

#visualizing the results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('BLUFF OR TRUTH(DECISION TREE REGRESSION)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
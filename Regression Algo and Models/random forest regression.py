#Random Forest Regression
#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#Fitting Random forest Regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

#Predicting the results
y_pred=regressor.predict(6.5)

#Visualizing the results
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('TRUTH OR BLUFF(RANDOM FOREST REGRESSION)')
plt.xlabel('LEVELS')
plt.ylabel('SALARY')
plt.show()
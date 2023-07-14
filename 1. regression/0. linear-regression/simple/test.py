import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import dump

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :1].values # experience
y = dataset.iloc[:, 1].values # salary

# we have no empty, nor do we need category nor stnadardization/normalization here

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
dump(regressor, 'simple_regression_salary_model.joblib')
y_pred = regressor.predict(X_test)

# Time to visualize

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Simple Regression Model: Salary Vs Years of Experience (Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Visualize (Test Set)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Simple Regression: Salary vs Expereince (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Predicted Salary")
plt.show()


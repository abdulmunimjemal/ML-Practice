from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :1].values # experience
y = dataset.iloc[:, 1].values # salary

# we have no empty, nor do we need category nor stnadardization/normalization here

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = load('simple_regression_salary_model.joblib')
y_pred = model.predict(X_test)

# Visualize (Test Set)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Simple Regression: Salary vs Expereince (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Predicted Salary")
plt.show()


# WORKED
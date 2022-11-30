import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

with open("Concrete_Data.csv") as file:
    data = pd.read_csv(file)


# Load the diabetes dataset
diabetes_X, diabetes_y = np.array(data.iloc[:, 0:7]), np.array(data.iloc[:, 8])

# Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 0]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# XY_test = pd.DataFrame({'x': diabetes_X_test[:, 0], 'y': diabetes_y_test})
# XY_pred = pd.DataFrame({'x': diabetes_X_test[:, 0], 'y': diabetes_y_pred})
#
# XY_test = XY_test.sort_values(by='x')
# XY_pred = XY_pred.sort_values(by='x')
#
# # чё за бред оно рисует
# plt.plot(XY_test['x'], XY_test['y'], color="black")
# plt.plot(XY_pred['x'], XY_pred['y'], color="blue", linewidth=3)
# plt.scatter(XY_test['x'], XY_test['y'], edgecolor='b', s=20, label="Samples")
#
#
#
# plt.show()
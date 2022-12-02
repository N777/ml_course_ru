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

results = []

for lasso in range(1, 11):
    for alpha in range(1, 11):
        # Create linear regression object
        enet = linear_model.ElasticNet(l1_ratio=lasso*0.1, alpha=alpha*0.1)

        # Train the model using the training sets
        enet.fit(diabetes_X_train, diabetes_y_train)

        # Make predictions using the testing set
        diabetes_y_pred = enet.predict(diabetes_X_test)

        mean_error = mean_squared_error(diabetes_y_test, diabetes_y_pred)
        score = r2_score(diabetes_y_test, diabetes_y_pred)
        need_print = False
        if need_print:
            print("l1_ratio: ", lasso * 0.1, "alpha: ", alpha * 0.1)
            # The coefficients
            print("Coefficients: \n", enet.coef_)
            # The mean squared error
            print("Mean squared error: %.2f" % mean_error)
            # The coefficient of determination: 1 is perfect prediction
            print("Coefficient of determination: %.2f" % score)
            print("--------------------------------------------------")
        results.append({"l1_ratio": lasso*0.1, "alpha": alpha*0.1, "mean_error": mean_error, "score": score})

results = pd.DataFrame(results)
print(results[results.score == results.score.max()])


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

with open("slump_test.data") as file:
    data = pd.read_csv(file)

# Load the diabetes dataset
diabetes_x, diabetes_y = np.array(data.iloc[:, 1:7]), np.array(data.iloc[:, 9])

test_count = 5
# Split the data into training/testing sets
diabetes_X_train = diabetes_x[:-test_count]
diabetes_X_test = diabetes_x[-test_count:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-test_count]
diabetes_y_test = diabetes_y[-test_count:]

results = []

for alpha in range(0, 910):
    # Create linear regression object
    enet = linear_model.Lasso(alpha=alpha)

    # Train the model using the training sets
    enet.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = enet.predict(diabetes_X_test)

    mean_error = mean_squared_error(diabetes_y_test, diabetes_y_pred)
    score = r2_score(diabetes_y_test, diabetes_y_pred)
    need_print = False
    if need_print:
        print("l1_ratio: ","alpha: ", alpha)
        # The coefficients
        print("Coefficients: \n", enet.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_error)
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % score)
        print("--------------------------------------------------")
    results.append({"alpha": alpha, "mean_error": mean_error, "score": score})

results = pd.DataFrame(results)
results.plot(x='alpha', y='score')
results.plot(x='alpha', y='mean_error')
plt.show()
print(results[results.score == results.score.max()])


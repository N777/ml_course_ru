import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

with open("Concrete_Data.csv") as file:
    data = pd.read_csv(file)

# Load the diabetes dataset
diabetes_x, diabetes_y = np.array(data.iloc[:, 0:7]), np.array(data.iloc[:, 8])
np.random.seed(0)

n_samples = 1010
degrees = [*range(1, 8)]

x = diabetes_x[:-20]
y = diabetes_y[:-20] + np.random.randn(n_samples) * 0.1
df_scores = []
plt.figure(figsize=(14, 5))
for num, i in enumerate(degrees):
    # ax = plt.subplot(1, len(degrees), num + 1)
    # plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=i,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(x, y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, x, y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = diabetes_x[-20:]

    score = r2_score(diabetes_y[-20:], pipeline.predict(X_test))
    df_scores.append({'degrees': i, 'score': score})

    # plt.plot(X_test[:, 0], pipeline.predict(X_test), label="Model")
    # plt.plot(X_test[:, 0], diabetes_y[-20:], label="True function")
    # plt.scatter(x[:, 0], y, edgecolor='b', s=2, label="Samples")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # margin = 50
    # plt.xlim(X_test.min() - margin, X_test.max() + margin)
    # plt.ylim(diabetes_y[-20:].min() - margin, diabetes_y[-20:].max() + margin)
    # plt.legend(loc="best")
    # plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(i, -scores.mean(), scores.std()))
df_scores = pd.DataFrame(df_scores)
df_scores.plot(x='degrees', y='score')
# plt.ylim(-1, df_scores['score'].max() + 0.2)
plt.show()

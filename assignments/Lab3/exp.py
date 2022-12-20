import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

with open("slump_test.data") as file:
    data = pd.read_csv(file)

# Load the diabetes dataset
diabetes_x, diabetes_y = np.array(data.iloc[:, 1:7]), np.array(data.iloc[:, 9])
np.random.seed(0)

# scaler = preprocessing.MinMaxScaler()
# diabetes_x = scaler.fit_transform(diabetes_x)

n_samples = diabetes_x.shape[0]
degrees = [*range(1, 10)]  # используем range тк при linspace генерит дробные числа
test_count = 5
x = diabetes_x[:-test_count]
y = diabetes_y[:-test_count] + np.random.randn(n_samples-test_count) * 0.1
test_x = diabetes_x[-test_count:]
test_y = diabetes_y[-test_count:]
df_scores = []
for num, i in enumerate(degrees):

    polynomial_features = PolynomialFeatures(degree=i,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(x, y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, x, y,
                             scoring="neg_mean_squared_error", cv=10)

    score_etalon = r2_score(y, pipeline.predict(x))
    score = r2_score(test_y, pipeline.predict(test_x))
    df_scores.append({'degrees': i, 'score_test': score, 'score_train': score_etalon})
    print(f"degrees: {i} score_train: {score_etalon} score_test: {score}")

df_scores = pd.DataFrame(df_scores)
plt.ylim(-0.1, 1.1)
plt.plot(df_scores['degrees'], df_scores['score_train'], "r", label='score_train')
plt.plot(df_scores['degrees'], df_scores['score_test'], "b", label='score_test')
plt.legend(['score_train', 'score_test'])
plt.show()

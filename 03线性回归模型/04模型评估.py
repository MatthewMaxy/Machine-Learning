import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # boston.data
y = raw_df.values[1::2, 2]  # boston.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

clf = LinearRegression()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)


# MSE 均方误差
def mse_score(y_test, y_predict):
	mse_score = 0.0
	for i in range(0, len(y_test)):
		mse_score += (y_test[i]-y_predict[i])**2
	return mse_score/len(y_test)


print(mse_score(y_test, y_predict))

# sklearn封装MSE
print("MSE: ", mean_squared_error(y_test, y_predict))

# RMSE 根均方误差
print("RMSE:", sqrt(mean_squared_error(y_test, y_predict)))

# MAE 平均绝对误差
print("MAE: ", mean_absolute_error(y_test, y_predict))

# score
print("score: ", clf.score(x_test, y_test))
# r2_score
print("r2_score: ", r2_score(y_test, y_predict))

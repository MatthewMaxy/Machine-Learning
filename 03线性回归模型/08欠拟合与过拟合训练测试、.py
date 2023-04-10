import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5*x**2 + x + np.random.normal(0, 1, size=100)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=666)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_predict = lin_reg.predict(x_test)
print(mean_squared_error(y_test, y_predict))

def pipeRegression(degree):
	return Pipeline([
		('poly', PolynomialFeatures(degree=degree)),
		('scaler', StandardScaler()),
		('lin_reg', LinearRegression())
	])

reg2 = pipeRegression(degree=2)
reg2.fit(x_train, y_train)
y_predict2 = reg2.predict(x_test)
print(mean_squared_error(y_test, y_predict2))


reg100 = pipeRegression(degree=100)
reg100.fit(x_train, y_train)
y_predict100 = reg100.predict(x_test)
print(mean_squared_error(y_test, y_predict100))


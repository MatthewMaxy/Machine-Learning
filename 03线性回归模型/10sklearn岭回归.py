from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def pipeRegression(degree, alpha):
	return Pipeline([
		('poly', PolynomialFeatures(degree=degree)),
		('scaler', StandardScaler()),
		('lin_reg', Ridge(alpha=alpha))
	])


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5*x**2 + x + np.random.normal(0, 1, size=100)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=666)
ridge1 = pipeRegression(50, 0.1)
ridge1.fit(x_train, y_train)

X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = ridge1.predict(X_plot)
plt.scatter(X, y)
plt.plot(X_plot[:, 0], y_plot, color='r')
plt.show()

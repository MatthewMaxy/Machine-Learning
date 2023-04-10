import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5*x**2 + x + np.random.normal(0, 1, size=100)
pipe_reg = Pipeline([
	('poly', PolynomialFeatures(degree=2)),
	('scaler', StandardScaler()),
	('lin_reg', LinearRegression())
])
pipe_reg.fit(X, y)
y_predict = pipe_reg.predict(X)
plt.scatter(x, y)
plt.scatter(x, y_predict)
plt.show()
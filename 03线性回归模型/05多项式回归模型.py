import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5*x**2 + x + np.random.normal(0, 1, size=100)
"""
一次关系
reg = LinearRegression()
reg.fit(X, y)
y_predict = reg.predict(X)
plt.scatter(x, y_predict)
plt.show()
"""
'''
手动二次回归
X2 = np.hstack([X, X**2])
reg2 = LinearRegression()
reg2.fit(X2, y)
y_predict2 = reg2.predict(X2)
plt.scatter(x, y)
plt.scatter(x, y_predict2)
plt.show()
'''
# sklearn实现多项式回归模型
poly = PolynomialFeatures(degree=3)
poly.fit(X)
X2 = poly.transform(X)
reg2 = LinearRegression()
reg2.fit(X2, y)
y_predict2 = reg2.predict(X2)
plt.scatter(x, y)
plt.scatter(x, y_predict2)
plt.show()

# 关于PolynomialFeatures
x = np.arange(1, 11).reshape(-1, 2)
print(x.shape)  # (5, 2)
poly = PolynomialFeatures(degree=3)
poly.fit(x)
x3 = poly.transform(x)
print(x3.shape)     # (5, 10)


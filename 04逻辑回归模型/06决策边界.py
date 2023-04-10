import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=61)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# def calculate_x2(x1):
# 	return (-1 * clf.intercept_ - clf.coef_[0][0]*x1)/clf.coef_[0][1]
#
# x1 = np.linspace(4, 8, 100)
# x2 = calculate_x2(x1)

# 定义绘制函数
def plot_decisionboundary(model, X):
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
	predict = model.predict(np.c_[xx.ravel(), yy.ravel()])
	predict = predict.reshape(xx.shape)
	from matplotlib.colors import ListedColormap
	plt.contourf(xx, yy, predict, cmap=ListedColormap(['#CCCCFF', '#EF9A9A', '#90CAF9']))


# 测试鸢尾花
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
X = iris.data[:, :2]
y = iris.target
knn_clf.fit(X, y)
plot_decisionboundary(knn_clf, X)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='g')
plt.show()

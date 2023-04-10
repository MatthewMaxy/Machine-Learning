import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def plot_decisionboundary(model, X):
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
	predict = model.predict(np.c_[xx.ravel(), yy.ravel()])
	predict = predict.reshape(xx.shape)
	from matplotlib.colors import ListedColormap
	plt.contourf(xx, yy, predict, cmap=ListedColormap(['#CCCCFF', '#EF9A9A', '#90CAF9']))


iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

# clf = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=60)
# clf.fit(X, y)
# plot_decisionboundary(clf, X)
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g')
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color='b')
# plt.show()

clf = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=60)
clf.fit(X, y)
plot_decisionboundary(clf, X)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='b')
plt.show()
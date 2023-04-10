import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = datasets.make_moons(noise=0.2, random_state=66)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

def plot_decisionboundary(model, X):
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
	predict = model.predict(np.c_[xx.ravel(), yy.ravel()])
	predict = predict.reshape(xx.shape)
	from matplotlib.colors import ListedColormap
	plt.contourf(xx, yy, predict, cmap=ListedColormap(['#CCCCFF', '#EF9A9A', '#90CAF9']))


plot_decisionboundary(dt_clf, X)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
print(dt_clf.score(X_test, y_test))

dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X_train, y_train)
plot_decisionboundary(dt_clf2, X)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
print(dt_clf2.score(X_test, y_test))
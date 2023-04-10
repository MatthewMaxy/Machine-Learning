import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

def get_gini(y):
	counter = Counter(y)
	gini = 1.0
	for value in counter.values():
		p = float(value / len(y))
		gini += -p**2
	return gini

def split_node(X, y, feature, value):
	left_arr = (X[:, feature] <= value)
	right_arr = (X[:, feature] > value)
	return X[left_arr], X[right_arr], y[left_arr], y[right_arr]

def split(X, y):
	best_gini = 1e6
	best_feature = -1
	best_value = -1
	for feature in range(X.shape[1]):
		sorted_index = np.argsort(X[:, feature])
		for i in range(1, len(X[:, feature])):
			if X[sorted_index[i-1], feature] != X[sorted_index[i], feature]:
				value = (X[sorted_index[i-1], feature] + X[sorted_index[i], feature])/2
				X_left, X_right, y_left, y_right = split_node(X, y, feature, value)
				g = len(X_left)/len(X)*get_gini(y_left) + len(X_right)/len(X)*get_gini(y_right)

				if g < best_gini:
					best_value = value
					best_feature = feature
					best_gini = g
	return best_gini, best_feature, best_value


best_gini, best_feature, best_value = split(X, y)
print('best_gini: ', best_gini)
print('best_feature: ', best_feature)
print('best_value: ', best_value)
X_left, X_right, y_left, y_right = split_node(X, y, best_feature, best_value)
print(get_gini(y_left))
print(get_gini(y_right))


best_gini, best_feature, best_value = split(X_right, y_right)
print('best_gini2: ', best_gini)
print('best_feature2: ', best_feature)
print('best_value2: ', best_value)
X_left, X_right, y_left, y_right = split_node(X_right, y_right, best_feature, best_value)
print(get_gini(y_left))
print(get_gini(y_right))

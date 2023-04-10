import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifer:
	def __init__(self, k):
		"""初始化分类器"""
		self._x_train = None
		self._y_train = None
		self.k = k

	def fit(self, x_train, y_train):
		self._x_train = x_train
		self._y_train = y_train
		return self

	def predict(self, x_predict):
		return [self._predict(x) for x in x_predict]

	def _predict(self, x):
		distance = []
		distance = [sqrt(np.sum((xi - x) ** 2)) for xi in self._x_train]
		nearst = np.argsort(distance)
		topK_y = [self._y_train[i] for i in nearst[:self.k]]
		count = Counter(topK_y)
		predict_y = count.most_common(1)[0][0]

		return predict_y


X = [[3.4, 2.8],
     [3.1, 1.8],
     [1.5, 3.4],
     [3.6, 4.7],
     [2.7, 2.9],
     [7.4, 4.5],
     [5.7, 3.5],
     [9.2, 2.5],
     [7.9, 3.4]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1]
x_train = np.array(X)
y_train = np.array(y)
clf = KNNClassifer(k=3)
clf.fit(x_train, y_train)
x = np.array([5.1, 3.4])
print(clf.predict(x.reshape(1, -1)))

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
from sklearn.linear_model import LogisticRegression

X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)


def TN(y_test, y_predict):
	return np.sum((y_test == 0) & (y_predict == 0))


def TP(y_test, y_predict):
	return np.sum((y_test == 1) & (y_predict == 1))


def FP(y_test, y_predict):
	return np.sum((y_test == 0) & (y_predict == 1))


def FN(y_test, y_predict):
	return np.sum((y_test == 1) & (y_predict == 0))


def precision(y_test, y_predict):
	return TP(y_test, y_predict) / (TP(y_test, y_predict) + FP(y_test, y_predict))


print(precision(y_test, y_predict))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
print(confusion_matrix(y_test, y_predict))
print(precision_score(y_test, y_predict))
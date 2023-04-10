import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = datasets.make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
'''
 集成学习手撸版
 
# KNN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))

# 逻辑回归
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
print(log_clf.score(X_test, y_test))

# 决策树
dt_clf = DecisionTreeClassifier(random_state=666)
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_test, y_test))

# 投票得出结果
y_predict1 = knn_clf.predict(X_test)
y_predict2 = log_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)
y_predict = (y_predict1 + y_predict2 + y_predict3) >= 2
y_predict = np.array(y_predict, dtype='int')

from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict, y_test))
'''

# 集成学习包
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
	estimators=[
		('knn_clf', KNeighborsClassifier(5)),
		('loh_clf', LogisticRegression()),
		('dt_clf', DecisionTreeClassifier(random_state=666))
	], voting='hard'
)
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))

# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()

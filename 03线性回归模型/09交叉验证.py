import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)
clf = KNeighborsClassifier()
'''
best_score = 0.0
best_k = 0.0
for k in range(1,10):
	knn_clf = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(clf, X_train, y_train)
	score = np.mean(scores)
	if best_score < score:
		best_k = k
		best_score = score
print(best_k, best_score)
'''
# 网格搜索
param_grid = [{
	'n_neighbors':[i for i in range(1, 11)]
}]

gridsearch = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
gridsearch.fit(X_train, y_train)
best_clf = gridsearch.best_estimator_
print(best_clf.score(X_test, y_test))

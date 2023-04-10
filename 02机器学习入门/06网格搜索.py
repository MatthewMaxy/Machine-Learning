# 网格搜索gridsearch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

clf = KNeighborsClassifier()
param_grid = [{
	'n_neighbors': [i for i in range(1, 10)],
	'weights': ['uniform', 'distance']
}]
gs_clf = GridSearchCV(clf, param_grid=param_grid)
gs_clf.fit(X_train, y_train)
print(gs_clf.best_params_, gs_clf.best_score_)
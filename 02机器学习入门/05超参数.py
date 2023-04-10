from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
best_weights = ""
best_score = 0.0
best_k = 0
for weight in ['uniform', 'distance']:
	for i in range(1, 10):
		clf = KNeighborsClassifier(n_neighbors=i, weights=weight)
		clf.fit(X_train, y_train)
		y_predict = clf.score(X_test, y_test)
		if y_predict > best_score:
			best_weights = weight
			best_score = y_predict
			best_k = i

print("best_k:", best_k)
print("best_score:", best_score)
print("best_weight:", best_weights)



from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

X, y = datasets.make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


voting_clf = VotingClassifier(
	estimators=[
		('knn_clf', KNeighborsClassifier(5)),
		('loh_clf', LogisticRegression()),
		('dt_clf', DecisionTreeClassifier(random_state=66))
	], voting='hard'
)

voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))

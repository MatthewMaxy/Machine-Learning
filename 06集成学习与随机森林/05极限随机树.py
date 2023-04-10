from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier
wine = load_wine()
X = wine.data
y = wine.target

et_clf = ExtraTreesClassifier(n_estimators=100, oob_score=True, random_state=666, bootstrap=True)
et_clf.fit(X, y)
print(et_clf.oob_score_)
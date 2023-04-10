from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
wine = load_wine()
X = wine.data
y = wine.target

rdf_clf = RandomForestClassifier(n_estimators=100, random_state=6, oob_score=True)
rdf_clf.fit(X, y)
print(rdf_clf.oob_score_)


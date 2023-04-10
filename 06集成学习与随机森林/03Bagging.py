from sklearn import datasets
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)


from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(DecisionTreeClassifier(random_state=666),
                        n_estimators=5000,
                        max_samples=150,
                        random_state=42, bootstrap=True, oob_score=True, n_jobs=12
                                )
start1 = time.perf_counter()
bagging_clf.fit(X_train, y_train)
end1 = time.perf_counter()
print(end1 - start1)
print(bagging_clf.score(X_test, y_test))
# # oob_score_ 用于测试
# bagging_clf.fit(X, y)
# print(bagging_clf.oob_score_)
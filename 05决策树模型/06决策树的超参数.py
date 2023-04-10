import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
wine = load_wine()
X = wine.data
y = wine.target

# print(X.shape)
# print(pd.Series(y).unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy',
                             random_state=30,
                             splitter='random',
                             max_depth=3,
                             # min_samples_leaf=10
                             min_samples_split=43,
                             max_leaf_nodes=5
                             )
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
feature_names = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', '稀释葡萄酒', '脯氨酸']
class_names =["拉菲", "雪莉", "巴贝拉"]
dot_data = tree.export_graphviz(clf,
                                feature_names=feature_names,
                                class_names=class_names,
                                filled=True,
                                rounded=True,
                                fontname='FangSong'
                                )
graph = graphviz.Source(dot_data)
graph.view()
print([*zip(feature_names, clf.feature_importances_)])
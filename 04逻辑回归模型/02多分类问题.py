import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

# ovr
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# ovo
model2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model2.fit(X_train, y_train)
print(model2.score(X_test, y_test))

# 导入ovo/ovr，可以改变分类原理（逻辑回归/随机森林……）
from sklearn.multiclass import OneVsOneClassifier
ovr = OneVsOneClassifier(LogisticRegression())
ovr.fit(X_train, y_train)
print(ovr.score(X_test, y_test))
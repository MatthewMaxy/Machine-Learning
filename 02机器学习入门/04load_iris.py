from sklearn import datasets

# 调用鸢尾花数据集测试
iris = datasets.load_iris()
X = iris.data
y = iris.target

# scikit-learn train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# KNN算法测试
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

# 自定义准确率函数
'''
def score(X_test, y_test):
	y_predict = clf.predict(X_test)
	return sum(y_predict == y_test)/len(y_test)
'''

# slearn metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))

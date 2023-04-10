from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

ss = StandardScaler()
ss.fit(X_train)
std_x_train = ss.transform(X_train)
# 注意：test数据集的归一化也应当基于训练集的归一化训练
std_x_test = ss.transform(X_test)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(std_x_train, y_train)
print(clf.score(std_x_test, y_test))

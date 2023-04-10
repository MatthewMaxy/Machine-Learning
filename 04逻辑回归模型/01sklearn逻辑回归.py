from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def polynomial_model(degree, C=0.1, penalty='l2', solver='lbfgs'):
	return Pipeline([
	('poly_feature', PolynomialFeatures(degree=degree)),
	('stand', StandardScaler()),
	('logist_regression', LogisticRegression(C=C, penalty=penalty, solver=solver))
])


model = polynomial_model(2, 2, 'l1', 'liblinear')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))



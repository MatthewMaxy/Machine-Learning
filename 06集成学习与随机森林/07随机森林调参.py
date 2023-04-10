from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
'''n_estimators'''
# scorel = []
# param_grid = {'n_estimators':np.arange(1, 200, 10)}
# rfc = RandomForestClassifier(n_jobs=-1, random_state=90)
# gridSearch0 = GridSearchCV(rfc, param_grid, cv=10)
# gridSearch0.fit(X, y)
# # {'n_estimators': 71} 0.9631265664160402

# best_score = 0.9631265664160402
# scorel = []
# param_grid = {'n_estimators':np.arange(60, 80, 1)}
# rfc = RandomForestClassifier(n_jobs=-1, random_state=90)
# gridSearch0 = GridSearchCV(rfc, param_grid, cv=10)
# gridSearch0.fit(X, y)
# print(gridSearch0.best_score_, gridSearch0.best_params_)
# # 0.9666353383458647 {'n_estimators': 73}

'''max_depth'''
# score1 = []
# for i in range(1, 20, 1):
# 	rfc = RandomForestClassifier(
# 		n_estimators=73, n_jobs=-1, random_state=90, max_depth=i)
# 	score = cross_val_score(rfc, X, y, cv=10).mean()
# 	score1.append(score)
# print(max(score1), score1.index(max(score1))+1)
# # 0.9666353383458647 8
# 发现欠拟合

'''max_features'''
# param_grid = {'max_features': np.arange(5, 30, 1)}
# rfc = RandomForestClassifier(n_estimators=73, n_jobs=-1, random_state=90)
# gridSearch0 = GridSearchCV(rfc, param_grid=param_grid, cv=10)
# gridSearch0.fit(X, y)
# print(gridSearch0.best_score_, gridSearch0.best_params_)
# # 0.9666666666666668 {'max_features': 24}

'''criterion'''
# param_grid = {'criterion': ['gini', 'entropy']}
# rfc = RandomForestClassifier(n_estimators=73, max_features=24, n_jobs=-1, random_state=90)
# gridSearch0 = GridSearchCV(rfc, param_grid=param_grid, cv=10)
# gridSearch0.fit(X, y)
# print(gridSearch0.best_score_, gridSearch0.best_params_)
# # 0.9666666666666668 {'criterion': 'gini'}

rfc = RandomForestClassifier(n_estimators=73, random_state=90, max_features=24, n_jobs=-1)
score = cross_val_score(rfc, cancer.data, cancer.target, cv=10).mean()
print(score)
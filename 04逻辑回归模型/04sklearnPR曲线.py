import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
digits = datasets.load_digits()


X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = model.decision_function(X_test)

# 自定义PR曲线
# thresholds = np.arange(np.min(scores), np.max(scores), 0.1)
# precisions = []
# recalls = []
# for threshold in thresholds:
# 	y_predict_1 = np.array(scores >= threshold, dtype='int')
# 	pre = precision_score(y_test, y_predict_1)
# 	rec = recall_score(y_test, y_predict_1)
# 	precisions.append(pre)
# 	recalls.append(rec)
#
# plt.plot(precisions, recalls)
# plt.show()

# 调包PR曲线
precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
plt.plot(precisions, recalls)
plt.show()
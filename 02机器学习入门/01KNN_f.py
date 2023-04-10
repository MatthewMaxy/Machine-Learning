import numpy as np
from math import sqrt
from collections import Counter
X = [[3.4, 2.8],
     [3.1, 1.8],
     [1.5, 3.4],
     [3.6, 4.7],
     [2.7, 2.9],
     [7.4, 4.5],
     [5.7, 3.5],
     [9.2, 2.5],
     [7.9, 3.4]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1]
x_train = np.array(X)
y_train = np.array(y)

x = np.array([5.1, 3.4])

# KNN过程

distance = []

# 计算每个点距离x距离
for xi in x_train:
	d = sqrt(np.sum((xi - x) ** 2))
	distance.append(d)

# np.argsort返回当前列表排序索引
nearst = np.argsort(distance)

# k为参与投票点数
k = 3
topK_y = [y_train[i] for i in nearst[:k]]
# Counter分类计数：Counter({1: 1, 0: 2})
count = Counter(topK_y)
# most_common(n) 为计数前n个 返回[(0,2)]
predict_y = count.most_common(1)[0][0]
print(predict_y)



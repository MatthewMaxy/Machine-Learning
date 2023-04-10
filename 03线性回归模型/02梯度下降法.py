import numpy as np
X = 2*np.random.rand(100, 1)
y = 3.0*X + 4 + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
eta = 0.1    # 学习率
n_iteration = 1000000 # 迭代次数
m = 100     # 样本数量
theta = np.random.randn(2, 1)
for iteration in range(n_iteration):
	gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
	theta = theta - eta * gradients

print(theta)

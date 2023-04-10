from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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
cls = KNeighborsClassifier(n_neighbors=3)
cls.fit(x_train, y_train)
x = np.array([5.1, 3.4])
print(cls.predict(x.reshape(1, -1)))
import numpy as np
import matplotlib.pyplot as plt

# 归一化操作
x = np.random.randint(0, 100, (50, 2))
x = np.array(x, dtype=float)
# print(np.mean(x[:, 0]), np.std(x[:, 0]))
x[:, 0] = (x[:, 0] - np.mean(x[:, 0]))/np.std(x[:, 0])
x[:, 1] = (x[:, 1] - np.mean(x[:, 1]))/np.std(x[:, 1])
plt.scatter(x[:, 0], x[:, 1])
print(np.mean(x[:, 0]), np.std(x[:, 0]))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def H(p):
	return -p * np.log(p) - (1 - p) * np.log(1 - p)


p = np.linspace(0.01, 0.99, 1000)
plt.plot(p, H(p))
plt.show()

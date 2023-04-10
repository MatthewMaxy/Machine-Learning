"""
子图
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1,100)
fig=plt.figure(figsize=(20,10),dpi=10)
ax1=fig.add_subplot(2,2,1)
ax1.plot(x,x)
ax2=fig.add_subplot(2,2,2)
ax2.plot(x,x**2)
plt.show()
"""
import random

'''
限制范围
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10,11,1)
y =x**2
plt.plot(x,y)
plt.xlim([-3,10])
plt.show()
'''

'''
坐标轴移动和改变：
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(0,14,2)
x=[-3,-2,-1,0,1,2,3]

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('blue')
ax.spines['left'].set_color('red')

ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',1))
plt.plot(x,y)
plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np
x = range(1,8)
y = [3,1,20,1,23,18,13]
size = np.random.randint(0,100,7)
plt.scatter(x,y,color='red',s=size)
plt.show()

"""import numpy as np
list1 = [[0,1,2],[4,2,3]]
one = np.array(list1)
print(one)
print(one.shape)
fi = one.reshape(3,2)
print(one)
print(fi)
"""
"""import numpy as np
array1 = np.array(range(24)).reshape(4,6)
print(array1)
print(np.insert(array1,2,[[1,2,3,4,5,6]],axis=0))
"""
import numpy as np

"""a = np.array([1,2,2,3,14,12,12,1])
print(a)
u, indices = np.unique(a, return_counts=True)
print(u, indices)"""
a = np.arange(24, dtype=float).reshape(4, 6)
a[3, 4], a[3, 5] = np.nan, np.nan
a[2, 5] = np.nan
print(a)
for i in range(a.shape[1]):
	list1 = a[:, i]
	num = np.count_nonzero(list1 != list1)
	if num != 0:
		list2 = list1[list1 == list1]
		list1[np.isnan(list1)] = np.mean(list2)
print(a)

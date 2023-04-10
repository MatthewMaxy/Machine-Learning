import pandas as pd
from pandas import Series,DataFrame
import numpy as np
# df1 = DataFrame(np.random.randint(0,10,(4,4)),index=[1,2,3,4],columns=list("abcd"))
# dict = {
# 	'Province':['GD','BJ', 'QH', 'FJ'],
# 	'pop': [1.3, 2.5, 1.1, 0.7],
# 	'year': [2018,2018,2018,2018]
# }
# df2 = DataFrame(dict,index=[1,2,3,4])
# dict = {"a":[1,2,3],"b":[4,5,6]}
# df2 = DataFrame.from_dict(dict)
# print(df2)
# dict = {
# 	'Province':pd.Series(['GD','BJ', 'QH', 'FJ'],index=list('abcd')),
# 	'pop': pd.Series([1.3, 2.5, 1.1],index=list('abc')),
# 	'year': pd.Series([2018,2018,2018],index=list('abd'))
# }
# def test(x):
# 	return "a"+x
#
# df = DataFrame.from_dict(dict)
# print(df)
# print(df.rename(index=test,columns=test,inplace=True))
# print(df)
# df3 =pd.DataFrame({"名字:['A','B','C'],"':[2,3,4]},index=list('abc'))
# df4 =pd.DataFrame({'Blue':[1,3,5],'Yellow':[2,3,4]},index=list('cbe'))
# print(df3)
# print(df4)
# print(df3.join(df4,how='outer'))

class1=['python','math','En',"C"]
class2=['期中','期末']
m_index = pd.MultiIndex.from_product([class2,class1])
df2=DataFrame(np.random.randint(0,150,(8,4)),index=m_index)
print(df2)
"""
sel = Series([1,2,3,4],index=list("abcd"))
print(sel)
print(sel['c'],sel[2])
print(sel['a':'c'])
"""
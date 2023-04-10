from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import graphviz
from sklearn import tree
# 数据导入与归一化
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # boston.data
y = raw_df.values[1::2, 2]   # boston.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=66)
dt_reg = DecisionTreeRegressor(max_depth=3)
dt_reg.fit(x_train, y_train)
dot_data = tree.export_graphviz(dt_reg,
                                filled=True,
                                rounded=True,
                                fontname='FangSong'
                                )
graph = graphviz.Source(dot_data)
graph.view()

print(dt_reg.score(x_test, y_test))
print(dt_reg.score(x_train, y_train))
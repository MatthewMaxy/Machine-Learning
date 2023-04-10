import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# 数据导入与归一化
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # boston.data
y = raw_df.values[1::2, 2]   # boston.target

reg = RandomForestRegressor(n_estimators=100, oob_score=True)
reg.fit(X, y)
print(reg.oob_score_)


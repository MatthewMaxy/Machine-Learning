import bls as bls
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
# 数据导入与归一化
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # boston.data
y = raw_df.values[1::2, 2]   # boston.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=66, test_size=0.2)
scaler = StandardScaler()
scaler.fit(x_train)
std_x_train = scaler.transform(x_train)
std_x_test = scaler.transform(x_test)

# 随机梯度下降法的线性规划 scikit learn SGD
sgd_reg = SGDRegressor()
sgd_reg.fit(std_x_train, y_train)
print(sgd_reg.score(std_x_test, y_test))

# 调参n_iter_no_change,连续n次无变化提前终止
sgd_reg = SGDRegressor(n_iter_no_change=50)
sgd_reg.fit(std_x_train, y_train)
joblib.dump(bls,"model1.pkl")
# print(sgd_reg.score(std_x_test, y_test))

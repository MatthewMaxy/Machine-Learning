import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # boston.data
y = raw_df.values[1::2, 2]   # boston.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=66, test_size=0.2)

scaler = StandardScaler()
scaler.fit(x_train)

std_x_train = scaler.transform(x_train)
std_x_test = scaler.transform(x_test)

lin_reg = LinearRegression()
lin_reg.fit(std_x_train, y_train)
print(lin_reg.score(std_x_test, y_test))


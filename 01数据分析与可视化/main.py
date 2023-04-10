# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# import numpy as np
#
# #线性
# def func_linear(x, a, b):
#     print(f'{a}x+{b}')
#     return a * x+ b
# #二次
# def func_poly_2(x, a, b, c):
#     return a*x*x + b*x + c
# #三次
# def func_poly_3(x, a, b, c , d):
#     return a*x*x*x + b*x*x + c*x + d
# #幂函数
# def func_power(x, a, b):
#     return x**a + b
# #指数函数
# def func_exp(x, a, b):
#     return a**x + b
#
# # 待拟合点
# xdata = [3.411222923,3.354016435,3.298697015,3.245172805,3.193357816,3.14317146,3.09453814,3.047386866,3.001650908,2.957267485]
# ydata = [6.215507694,6.045479135,5.877595716,5.713402718,5.559527119,5.412984442,5.265277512,5.125748101,4.987707789,4.851639563]
#
# x = list(np.arange(2.8, 3.5, 0.001))
#
# # 绘制散点
# plt.scatter(xdata[:], ydata[:], 25, "red")
#
# # popt数组中，存放的就是待求的参数a,b,c,......
# popt, pcov = curve_fit(func_linear, xdata, ydata)
# y1 = [func_linear(i, popt[0], popt[1]) for i in x]
# plt.plot(x, y1, 'r')
#
#
# popt, pcov = curve_fit(func_poly_2, xdata, ydata)
# y2 = [func_poly_2(i, popt[0], popt[1], popt[2] ) for i in x]
# plt.plot(x, y2, 'g')
#
# popt, pcov = curve_fit(func_poly_3, xdata, ydata)
# y3 = [func_poly_3(i, popt[0], popt[1], popt[2] ,popt[3]) for i in x]
# plt.plot(x, y3, 'b')
#
# popt, pcov = curve_fit(func_power, xdata, ydata)
# y4 = [func_power(i, popt[0], popt[1]) for i in x]
# plt.plot(x, y4, 'y')
#
# popt, pcov = curve_fit(func_exp, xdata, ydata)
# y5 = [func_exp(i, popt[0], popt[1]) for i in x]
# plt.plot(x, y5, 'c')
#
# plt.show()

import math
print(math.ceil())
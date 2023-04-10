import matplotlib.pyplot as plt

x = range(1,8)
y = [3,1,20,1,23,18,13]
y2 = [1,2,2,23,23,23,7]

# 设置图片大小 figsize=width(pt),height(pt) 分辨率dpi
plt.figure(figsize=(10,8),dpi=80)

# 设置刻度个数、显示格式、倾斜
# plt.xticks(x)
# plt.xticks(range(1,7))
x_tick_label = ["{}.00".format(i) for i in x]
plt.xticks(x, x_tick_label, rotation=45)

y_tick_label = ["{}℃".format(i) for i in range(min(y), max(y)+1, 3)]
plt.yticks(range(min(y), max(y)+1, 3), y_tick_label)

'''
折线图设计
alpha:透明度，color:颜色, linewidth:线宽
linestyle:
- 实线    --短线    -.点横线   :点虚线
marker:数据点样式  markersize:数据点标记大小  markeredgecolor:数据点颜色   markeredgewidth:数据点边宽
'''
plt.plot(x,y,color='red',alpha=0.3, linestyle='--',linewidth=3,
         marker='^',markersize=5,markeredgecolor='green',markeredgewidth=5,label='南京')

# 两线
plt.plot(x,y2,color='brown', label='淮安')

# 绘制网格
plt.grid(alpha=0.3)

# 图片标题(设置中文标题需注意）
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\STKAITI.TTF',size=10)
plt.title("图片", fontproperties=my_font)

# 图例（中文问题）
# 位置loc:upper left  upper right  center left  upper center ……
plt.legend(prop=my_font, loc='center right')

# 坐标轴标题：
plt.xlabel('时间', fontproperties=my_font)
plt.ylabel('温度', fontproperties=my_font)

# 图片保存,放在show()前或不要show
plt.savefig('./t1.svg')

# show()会释放资源
plt.show()


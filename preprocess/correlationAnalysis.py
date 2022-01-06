# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 16:25
# @Author  : hujinghua
# @File    : correlationAnalysis.py
# @Software: IntelliJ IDEA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1、计算质检指标的相关系数矩阵
# plt.rc("font", family = 'KaiTi')    # 设置plt.show()的字体为楷体
# min_max_data = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 7)  #读取标准化后的数据表
# columns_names = min_max_data.columns  # 读取excel文件中的列名，给后续的图里的列和行配名字
# result = pd.DataFrame(np.corrcoef(min_max_data.T).round(3), columns=columns_names, index = columns_names)   # 计算皮尔森积矩相关系数矩阵
# result_abs = abs(result)
# plt.figure(figsize=(15, 15))
# sns.heatmap(result_abs, cmap = 'Blues', annot = True, square = True)    # 绘制热力图
# plt.figure(figsize = (15, 15))
# plt.savefig(r'C:\Users\Carrot\Desktop\picture.png', bbox_inches = 'tight', dpi = 600, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
# plt.show()
# # result.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx')

# 2、查看强相关性下的指标散点图
sns.set_style("whitegrid")
plt.rc("font", family = 'KaiTi')    # 设置plt.show()的字体为楷体
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
data_1 = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', sheet_name = "标准化后的质检指标数据")  #读取标准化后的数据表
y = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', sheet_name = "标准化后的质检指标数据", usecols = [1])  #读取标准化后的数据表
# fig = plt.figure()
# plt.set_title('室温下的屈服强度2——室温下的抗拉强度2')
# plt.set_xlabel('650℃下的抗拉强度2')
# plt.set_ylabel('650℃下的断面收缩率2')
# plt.scatter(x, y, s = 100, alpha = 0.75)
# plt.savefig(r'C:\Users\Carrot\Desktop\picture1.png', bbox_inches = 'tight', dpi = 300, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
# plt.show()

# 3、绘制强相关性下的线性回归以及散点图
# sns.lmplot(x = '室温下的屈服强度2', y = '室温下的抗拉强度2', data = data_1)
# sns.lmplot(x = '室温下的屈服强度2', y = '室温下的伸长率2', data = data_1)
# sns.lmplot(x = '室温下的抗拉强度2', y = '室温下的伸长率2', data = data_1)
sns.lmplot(x = '650℃下的抗拉强度2', y = '650℃下的断面收缩率2', data = data_1)
# sns.lmplot(x = '650℃下的伸长率2', y = '650℃下的断面收缩率2', data = data_1)
plt.savefig(r'C:\Users\Carrot\Desktop\650℃下的抗拉强度2——650℃下的断面收缩率2.png', bbox_inches = 'tight', dpi = 300, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
plt.show()
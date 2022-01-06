# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 15:14
# @Author  : hujinghua
# @File    : dataDistribute.py
# @Software: IntelliJ IDEA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def precoess(data, process_way):
    min_max = MinMaxScaler()  # 实例化min-max函数，实现归一化
    z_score = StandardScaler()  # 实例化z-score函数，实现标准化
    if process_way == 'min_max':
        data_1 = pd.DataFrame(min_max.fit_transform(data), columns=data.columns)  #对数据进行min_max归一化
    else :
        data_1 = pd.DataFrame(z_score.fit_transform(data), columns=data.columns)  #对数据进行标准化
    return data_1

# data_1.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx')  # 将归一化后的文件保存到Excel中
# distribute = data_1.describe()
# distribute.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx', sheet_name="distribute")  #输出到excel文件中

def plot(data_1, is_x):
    plt.rc("font", family = 'KaiTi')    # 设置plt.show()的字体为楷体
    plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
    if is_x:
        data_1.hist(figsize = (20, 10), bins = 20, sharex = is_x, layout = (3, 5))   # 绘制x轴相同的直方图
    else :
        data_1.hist(figsize = (20, 10), bins = 20, sharex = is_x, layout = (3, 5))   # 绘制大小为(20,20)的直方图
    plt.savefig(r'C:\Users\Carrot\Desktop\picture1.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)  # 让describe()的结果全部可见
    data = pd.read_csv(r'C:\Users\Carrot\Desktop\forestfires_data.csv')  #读取数据集
    data_1 = precoess(data, process_way = 'min_max') # min_max为归一化，z_score为标准化
    # plot(data_1, False)
    data_1.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx')  # 将归一化后的文件保存到Excel中
    # distribute = data_1.describe()
    # distribute.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx', sheet_name="distribute")  #输出到excel文件中


# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 10:04
# @Author  : hujinghua
# @File    : dpca.py
# @Software: IntelliJ IDEA
# 使用DPCA聚类算法来对质检指标进行聚类

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from scipy.spatial.distance import cdist

def get_point_density(datas, labels, min_distance, points_number):
    distance_all = np.random.rand(points_number, points_number)
    point_density = np.random.rand(points_number)

    # 计算得到各点间距离
    distance_all = cdist(datas, datas, metric='euclidean')
    # print('距离数组:\n',distance_all,'\n')

    # 计算得到各点的点密度，找到与第i个数据点之间的距离小于截断距离dc的数据点的个数，并将其作为第i个数据点真的密度
    for i in range(points_number):
        x = 0
        for n in range(points_number):
            if distance_all[i][n] > 0 and distance_all[i][n] <= min_distance:
                x = x + 1
            point_density[i] = x
    print('点密度数组:', point_density, '\n')
    return distance_all, point_density

def get_each_distance(distance_all, point_density, data, laber):
    nn = [] # 每个样本点的聚类中心距离
    for i in range(points_number):
        aa = [] #大于自身点密度的索引数组
        for n in range(points_number):
            if (point_density[i] < point_density[n]):
                aa.append(n)
        print("大于自身点密度的索引", aa)
        ll = get_min_distance(aa, i, distance_all, point_density, data, laber)
        nn.append(ll)
    print(nn)
    return nn

# 得到到点密度大于自身的最近点的距离
def get_min_distance(aa, i, distance_all, point_density, data, laber):
    min_distance = []
    # 如果传入的aa为空，说明该点为点密度最大的点，该点的聚类中心距离计算方法与其他不同
    if aa != []:
        for k in aa:
            min_distance.append(distance_all[i][k])
        return min(min_distance)
    else:
        max_distance = get_max_distance(distance_all, point_density, laber)
        return max_distance

def get_max_distance(distance_all, point_density, laber):
    point_density = point_density.tolist()
    a = int(max(point_density)) #最大点密度
    b = laber[point_density.index(a)]   #最大点密度对应的索引
    c = max(distance_all[b])    #最大点密度对应的聚类中心距离
    return c

def get_picture(data,laber,points_number,point_density,nn):
    # 创建Figure
    fig = plt.figure(figsize=(15,5))
    # 用来正常显示中文标签
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    # 用来正常显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False

    # fig1. 绘制聚类后的数据分布图
    # 给nn进行归一化
    max_1 = max(nn)
    min_1 = min(nn)
    for i in range(len(nn)):
        nn[i] = (nn[i] - min_1) / (max_1 - min_1)
    # 聚类后分布
    ax1 = fig.add_subplot(121)
    plt.scatter(point_density.tolist(), nn ,c=laber, linewidths = 2.5, s = 150)
    plt.xlabel("局部密度")
    plt.ylabel("聚类中心距离")
    plt.title(u'聚类后数据分布')
    plt.sca(ax1)
    for i in range(points_number):
        plt.text(point_density[i],nn[i],laber[i], fontsize = 12, horizontalalignment = 'center')

    # fig2. 绘制聚类点偏离度值
    ax2 = fig.add_subplot(122)
    r = []  # r代表簇中心权值
    for i in range(points_number):
        r.append(nn[i] * point_density[i])
    r.sort()
    # print(temp)
    #[1.792, 0.0, 1.146, 1.28, 0.0, 9.0, 1.23, 1.575, 0.056, 2.14, 0.0, 1.024, 1.64]
    #[0.0, 0.0, 0.0, 0.056, 1.024, 1.146, 1.23, 1.28, 1.575, 1.64, 1.792, 2.14, 9.0]
    x_1 = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    plt.plot(x_1, r, linestyle = '--', color = 'k', linewidth = 2)
    plt.scatter(x_1, r, c = x_1, s = 100)
    plt.xlabel("聚类点")
    plt.ylabel("簇中心权值")
    plt.title('聚类点偏离度')
    plt.sca(ax2)
    #[1, 4, 10, 8, 11, 2, 6, 3, 7, 12, 0, 9, 5]
    label = [1, 4, 10, 8, 11, 2, 6, 3, 7, 12, 0, 9, 5]
    for i in range(points_number):
        plt.text(x_1[i], r[i], label[i], fontsize = 12, horizontalalignment = 'left')
    plt.savefig(r'C:\Users\Carrot\Desktop\picture.png', bbox_inches = 'tight', dpi = 600, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
    plt.show()

    plt.figure(figsize=(10, 10))
    y_2 = []
    for i in range(points_number):
        y_2.append(r[i])


if __name__ == '__main__':
    data = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 7).T
    laber = []  #laber代表指标的标签
    for i in range(0, 13):
        laber.append(i)
    min_distance = 4.6           # 邻域半径的设定看各点的距离
    points_number = 13          # 随机点个数

    # 计算各点间距离、各点点密度(局部密度)大小
    distance_all, point_density = get_point_density(data, laber, min_distance, points_number)
    # # 得到各点的聚类中心距离
    nn = get_each_distance(distance_all, point_density, data, laber)
    print('最后的各点密度：', point_density.tolist())
    max_1 = max(nn)
    min_1 = min(nn)
    for i in range(len(nn)):
        nn[i] = ((nn[i] - min_1) / (max_1 - min_1)).round(3)
    print('最后的各点中心距离：', nn)
    #
    # # 画图
    get_picture(data, laber, points_number, point_density, nn)
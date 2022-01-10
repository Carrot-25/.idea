# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 21:13
# @Author  : hujinghua
# @File    : apaAgorithm.py
# @Software: IntelliJ IDEA
import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 1、设置实验数据
def init_sample():
    centers = pd.read_excel(r'C:\Users\Carrot\Desktop\相关系数矩阵.xlsx', 2)
    dataLen = len(centers)
    return centers, dataLen
# 2、计算相似度矩阵，并且设置参考度，参考度为相似度矩阵的中值
def cal_simi(Xn, dataLen):
    min_max_data = pd.read_excel(r'C:\Users\Carrot\Desktop\相关系数矩阵.xlsx', 3).T  # 读取相似度矩阵
    simi = min_max_data.values
    p = np.min(simi)
    for i in range(dataLen):
        simi[i][i] = p
    return simi
# 3、计算吸引度矩阵，即R值
def init_R(dataLen):
    R = [[0] * dataLen for j in range(dataLen)]
    return R
def init_A(dataLen):
    A = [[0] * dataLen for j in range(dataLen)]
    return A
# 4、迭代更新R矩阵
def iter_update_R(dataLen,R,A,simi):
    old_r = 0 ##更新前的某个r值
    lam = 0.6 ##阻尼系数,用于算法收敛
    ##此循环更新R矩阵
    for i in range(dataLen):
        for k in range(dataLen):
            old_r = R[i][k]
            if i != k:
                max1 = A[i][0] + R[i][0]  ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if A[i][j] + R[i][j] > max1 :
                            max1 = A[i][j] + R[i][j]
                ##更新后的R[i][k]值
                R[i][k] = simi[i][k] - max1
                ##带入阻尼系数重新更新
                R[i][k] = (1-lam)*R[i][k] +lam*old_r
            else:
                max2 = simi[i][0] ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if simi[i][j] > max2:
                            max2 = simi[i][j]
                ##更新后的R[i][k]值
                R[i][k] = simi[i][k] - max2
                ##带入阻尼系数重新更新
                R[i][k] = (1-lam)*R[i][k] +lam*old_r
    print("max_r:"+str(np.max(R)))
    #print(np.min(R))
    return R
# 5、迭代更新A矩阵
def iter_update_A(dataLen,R,A):
    old_a = 0 ##更新前的某个a值
    lam = 0.6 ##阻尼系数,用于算法收敛
    ##此循环更新A矩阵
    for i in range(dataLen):
        for k in range(dataLen):
            old_a = A[i][k]
            if i ==k :
                max3 = R[0][k] ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if R[j][k] > 0:
                            max3 += R[j][k]
                        else :
                            max3 += 0
                A[i][k] = max3
                ##带入阻尼系数更新A值
                A[i][k] = (1-lam)*A[i][k] +lam*old_a
            else :
                max4 = R[0][k] ##注意初始值的设置
                for j in range(dataLen):
                    ##上图公式中的i!=k 的求和部分
                    if j != k and j != i:
                        if R[j][k] > 0:
                            max4 += R[j][k]
                        else :
                            max4 += 0

                ##上图公式中的min部分
                if R[k][k] + max4 > 0:
                    A[i][k] = 0
                else :
                    A[i][k] = R[k][k] + max4

                ##带入阻尼系数更新A值
                A[i][k] = (1-lam)*A[i][k] +lam*old_a
    print("max_a:"+str(np.max(A)))
    #print(np.min(A))
    return A
# 6、计算聚类中心
def cal_cls_center(dataLen,simi,R,A):
    ##进行聚类，不断迭代直到预设的迭代次数或者判断comp_cnt次后聚类中心不再变化
    max_iter = 1000    ##最大迭代次数
    curr_iter = 0     ##当前迭代次数
    max_comp = 1000     ##最大比较次数
    curr_comp = 0     ##当前比较次数
    class_cen = []    ##聚类中心列表，存储的是数据点在Xn中的索引
    while True:
        ##计算R矩阵
        R = iter_update_R(dataLen,R,A,simi)
        ##计算A矩阵
        A = iter_update_A(dataLen,R,A)
        ##开始计算聚类中心
        for k in range(dataLen):
            if R[k][k] +A[k][k] > 0:
                if k not in class_cen:
                    class_cen.append(k)
                else:
                    curr_comp += 1
        curr_iter += 1
        print(curr_iter)
        if curr_iter >= max_iter or curr_comp > max_comp :
            break
    return class_cen


if __name__=='__main__':
    #初始化数据
    Xn, dataLen = init_sample()
    ##初始化R、A矩阵
    R = init_R(dataLen)
    A = init_A(dataLen)
    ##计算相似度
    simi = cal_simi(Xn, dataLen)
    ##输出聚类中心
    class_cen = cal_cls_center(dataLen,simi,R,A)
    print(class_cen)
    # for i in class_cen:
    #    print(str(i)+":"+str(Xn[i]))
    # print(class_cen)

    ##根据聚类中心划分数据
    # c_list = []
    # for m in Xn:
    #     temp = pd.read_excel(r'C:\Users\Carrot\Desktop\相关系数矩阵.xlsx', 2).T  #读取标准化后的数据表
    #     temp = temp.values
    #     print(temp[0])
    #     ##按照是第几个数字作为聚类中心进行分类标识
    #     c = class_cen[temp.tolist().index((np.max(temp)))]
    #     c_list.append(c)
    # # ##画图
    # colors = ['red','blue','black','green','yellow']
    # plt.figure(figsize=(8,6))
    # plt.xlim([-3,3])
    # plt.ylim([-3,3])
    # for i in range(dataLen):
    #     d1 = Xn[i]
    #     d2 = Xn[c_list[i]]
    #     c = class_cen.index(c_list[i])
    #     plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
    #     #if i == c_list[i] :
    #     #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=3)
    #     #else :
    #     #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=1)
    # plt.show()
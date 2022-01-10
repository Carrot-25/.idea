# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 15:52
# @Author  : hujinghua
# @File    : apaAgorithm1.py
# @Software: IntelliJ IDEA
# from sklearn.cluster import AffinityPropagation
# from sklearn import metrics
# import numpy as np
# import pandas as pd
#
#
# #生成数据
# centers = pd.read_excel(r'C:\Users\Carrot\Desktop\相关系数矩阵.xlsx', 2).T
# dataLen = len(centers)
#
# min_max_data = pd.read_excel(r'C:\Users\Carrot\Desktop\相关系数矩阵.xlsx', 3)  # 读取相似度矩阵
# simi = min_max_data.values
# p = np.median(simi)
# for i in range(dataLen):
#     simi[i][i] = p
# print(pd.DataFrame(simi))
# # # p=-50   ##3个中心
# # p = np.min(simi)  ##9个中心，
# # #p = np.median(simi)  ##13个中心
# #
# ap = AffinityPropagation(damping=0.5,max_iter=500,convergence_iter=30,
#                          preference=p).fit(centers)
# cluster_centers_indices = ap.cluster_centers_indices_
# print(cluster_centers_indices)
# print(ap.cluster_centers_)
# print(ap.labels_)
#
# for idx in cluster_centers_indices:
#     print(centers[idx])

array = [["室温下的屈服强度2",2,3], ["室温试样硬度2",5,6]]
print(str(array[0]))
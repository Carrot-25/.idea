# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 14:28
# @Author  : hujinghua
# @File    : tsne.py
# @Software: IntelliJ IDEA

# 使用T-SNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import seaborn as sns

X = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 3)  #读取数据集
tsne = manifold.TSNE(n_components = 2, # 嵌入空间的维度
                     perplexity = 20, # 混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
                     early_exaggeration = 20,   # 嵌入空间簇间距的大小，默认为12，该值越大，可视化后的簇间距越大
                     learning_rate = 200,   # 学习率，表示梯度下降的快慢，默认为200，建议取值在10到1000之间
                     n_iter = 1000, # 迭代次数，默认为1000，自定义设置时应保证大于250
                     metric = "euclidean",  # 表示向量间距离度量的方式，默认是欧氏距离
                     init = 'pca',  #初始化，默认为random。取值为random为随机初始化，取值为pca为利用PCA进行初始化（常用）
                     verbose = 1,   #是否打印优化信息，取值0或1，
                     random_state = 501    #随机数种子，整数或RandomState对象
                     )
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

#嵌入空间可视化
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# palette = sns.color_palette("bright", 10)
labels = X.columns
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
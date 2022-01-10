# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 17:38
# @Author  : hujinghua
# @File    : pca.py
# @Software: IntelliJ IDEA

#使用pca来进行降维
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 5) #读取数据集
data_name = data.iloc[:, 0]
data_1 = data.iloc[:, 1:]
print(data_name)
print(data_1)
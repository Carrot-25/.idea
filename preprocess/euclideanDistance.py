# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 10:07
# @Author  : hujinghua
# @File    : euclideanDistance.py
# @Software: IntelliJ IDEA
import numpy as np
import pandas as pd

matrix_1 = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 6)
matrix_2 = matrix_1

def compute_distances_two_loop(test_matrix, train_matrix):
    num_test = test_matrix.shape[0]
    num_train = train_matrix.shape[0]
    dists = np.zeros((num_test, num_train))    # shape(num_test, num-train)
    for i in range(num_test):
        for j in range(num_train):
            # corresponding element in Numpy Array can compute directly,such as plus, multiply
            print(test_matrix.iloc(i))
            print(train_matrix.iloc(j))
            dists[i][j] = np.sqrt(np.sum(np.square(test_matrix.iloc(i) - train_matrix.iloc(j))))
            # print(test_matrix.iloc(i))
            # print(train_matrix.iloc(j))
    return dists

if __name__ == '__main__':
    data_column = pd.read_excel(r'C:\Users\Carrot\Desktop\发动机锻件预处理数据集.xlsx', 4)
    data = pd.DataFrame(compute_distances_two_loop(matrix_1, matrix_2), columns = data_column.columns, index = data_column.columns)
    print(data)
    # data.to_excel(r'C:\Users\Carrot\Desktop\data2.xlsx')

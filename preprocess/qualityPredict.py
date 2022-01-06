# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 20:23
# @Author  : hujinghua
# @File    : qualityPredict.py
# @Software: IntelliJ IDEA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import tree, svm, neighbors, ensemble
from sklearn.ensemble import BaggingRegressor

# 1、实例化min-max函数，实现归一化，并输出到excel文件中
def minmax():
    data = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', 0)  #读取数据集
    min_max = MinMaxScaler()  # 实例化min-max函数，实现归一化
    data_1 = pd.DataFrame(min_max.fit_transform(data), columns=data.columns)  #对数据进行min_max归一化
    data_1.to_excel(r'C:\Users\Carrot\Desktop\data1.xlsx')  # 将归一化后的文件保存到Excel中

#2、使用协方差计算相关系数
def calculate_data():
    data_x = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', sheet_name = '归一化后')  # corrcoef函数中已经进行了归一化处理
    columns_names = data_x.columns  # 读取excel文件中的列名，给后续的图里的列和行配名字
    # result = pd.DataFrame(np.corrcoef(data_x.T).round(3), columns=columns_names, index = columns_names)   # 计算皮尔森积矩相关系数矩阵
    result = data_x.corr('spearman')    #计算Spearman相关系数，主要反映非线性的
    # result = abs(result.iloc[-1:])  #选取DataFrame的最后一行
    print(result)
    result.to_excel(r'C:\Users\Carrot\Desktop\data2.xlsx', sheet_name = "Sheet2")


# 3、绘制相关系数的散点图
def plot_scatter(type):
    data = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', sheet_name = '{}'.format(type), usecols="A:AR")
    x = []
    for i in range(1, 45):
        x.append(i)
    data_1 = data.iloc[56]
    plt.Figure(figsize=(15, 15))
    plt.scatter(x, data_1.values)
    plt.savefig(r'C:\Users\Carrot\Desktop\picture1.png', bbox_inches = 'tight', dpi = 300, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
    plt.show()


#4、画出单因素拟合情况
def fitting():
    pd_data = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', sheet_name = '工艺参数数据集')#原始数表
    plt.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号，如果是plt画图，则将mlp换成plt
    print(pd_data.columns)
    # sns.pairplot(pd_data, x_vars=['C', 'Mn', 'Si', 'P', 'Cr', 'Ni', 'Mo', 'Cu', 'Al', 'Ti', 'O', 'N'],
    #              y_vars='室温下的抗拉强度2')
    # sns.pairplot(pd_data, x_vars=['Nb', 'Co', 'Mg', 'B', 'Ca', 'Pb', 'Sn', 'Se', 'Bi', 'Ag', 'Tl'],
    #              y_vars='室温下的抗拉强度2')
    sns.pairplot(pd_data, x_vars=['镦饼过程的终锻温度', '饼坯高度', '模锻转运时间', '模锻始锻温度',
                                  '锤击次数', '模锻终锻温度', '锻造时间', '欠压值'],
                 y_vars='室温下的抗拉强度2')
    plt.savefig(r'C:\Users\Carrot\Desktop\picture1.png', bbox_inches = 'tight', dpi = 300, pad_inches = 0)   # 保存热力图到本地，并设置紧凑，分辨率等
    plt.show()

#5、使用机器学习算法来预测
def build_lr(x, y, name) :
    plt.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
    plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 532)  # 选择30%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(x_train.shape,
                                                                                              y_train.shape,
                                                                                              x_test.shape,
                                                                                              y_test.shape))
    linreg_1 = LinearRegression()   #线性回归
    linreg_2 = tree.DecisionTreeRegressor()   #决策树回归
    linreg_3 = svm.SVR()  #SVM回归
    linreg_4 = neighbors.KNeighborsRegressor()    #KNN回归
    linreg_5 = ensemble.RandomForestRegressor(n_estimators=20)    #随机森林回归
    linreg_6 = ensemble.AdaBoostRegressor(n_estimators = 50) #Adaboost回归
    linreg_7 = ensemble.GradientBoostingRegressor(n_estimators = 100) #GBRT回归
    linreg_8 = BaggingRegressor() #Bagging回归
    list = [linreg_1, linreg_2, linreg_3, linreg_4, linreg_5, linreg_6, linreg_7, linreg_8]
    for i in range(0, 8):
        print("-----------------------------------------------------------------")
        print(list[i])
        #训练
        linreg = list[i]
        model = linreg.fit(x_train, y_train)
        #预测
        y_pred = linreg.predict(x_test)
        MSE = mean_squared_error(y_test, y_pred)
        MAE = mean_absolute_error(y_test, y_pred)
        R2 = cross_val_score(linreg, y_test, y_pred, cv = 5).mean() #使用交叉验证，其中k取5
        print("均方差MSE为：", MSE)
        print("平均绝对误差MAE为：", MAE)
        print("R2为：", R2)
        if (list[i] == linreg_5):
            print("特征变量重要性", model.feature_importances_)
        #做ROC曲线
        plt.plot(range(len(y_pred)), y_pred, 'b', marker = 'o', label="predict", ls = '--')
        plt.plot(range(len(y_pred)), y_test, 'r', marker = 'o', label="test", ls = '--')
        plt.legend(loc="upper right")  # 显示图中的标签
        plt.xlabel("样本点")
        plt.ylabel("模锻终锻温度")
        # if name == 1:
            # plt.savefig(r'C:\Users\Carrot\Desktop\{}——模锻终锻温度未进行参数约简.png'.format(linreg), bbox_inches = 'tight', dpi = 300, pad_inches = 0)
        # else :
            # plt.savefig(r'C:\Users\Carrot\Desktop\{}——模锻终锻温度进行参数约简.png'.format(linreg), bbox_inches = 'tight', dpi = 300, pad_inches = 0)
        plt.show()
    return
if __name__ == '__main__':
    # calculate_data()    #使用协方差计算相关系数，有person和spearman
    # str = "spearman"
    # plot_scatter(str)    #根据相关系数绘制散点图，有person和spearman
    # 未进行约简参数的
    x_1 = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="E:H").values#原始数表
    y_1 = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="I").values#原始数表
    name_1 = 1
    #进行约简参数的
    # x_2 = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', sheet_name = 'Sheet3', usecols = "A:Q")#原始数表
    # y_2 = pd.read_excel(r'C:\Users\Carrot\Desktop\质量预测数据集.xlsx', sheet_name = 'Sheet3', usecols = "R")#原始数表
    # name_2 = 2
    build_lr(x_1, y_1, name_1)
    # build_lr(x_2, y_2, name_2)
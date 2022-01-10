# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 19:44
# @Author  : hujinghua
# @File    : featureSelection.py
# @Software: IntelliJ IDEA

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import r2_score
from collections import defaultdict
# from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# boston = load_boston()
# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]

X1 = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="E:N")
Y2 = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="O")
X = X1.values
Y = Y2.values
names = X1.columns


# 单变量特征选择
def model_based_ranking():
    rf = RandomForestClassifier(n_estimators = 20, max_depth = 4)
    scores = []
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                cv=ShuffleSplit(len(X), 3, .3))
        scores.append((round(np.mean(score), 3), names[i]))
    print(sorted(scores, reverse=True))

# 随机森林——平均不纯度减少
def mean_decrease_impurity():
    rf = RandomForestClassifier()
    rf.fit(X, Y)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))

# 随机森林——平均精确率减少
def mean_decrease_accuracy():
    rf = RandomForestClassifier()
    scores = defaultdict(list)
    rs = ShuffleSplit(n_splits = len(X), test_size = 0.3)
    for train_idx, test_idx in rs.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True))

# 稳定性选择
def stability_selection():
    rlasso = RandomizedLasso

# 递归特征消除RFE
def recursive_feature_elimination():
    lr = RandomForestClassifier()
    rfe = RFE(lr, n_features_to_select = 1)
    rfe.fit(X, Y)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

if __name__ == '__main__':
    print("model_based_ranking")
    model_based_ranking()
    print("mean_decrease_impurity")
    mean_decrease_impurity()
    print("mean_decrease_accuracy")
    mean_decrease_accuracy()
    print("recursive_feature_elimination")
    recursive_feature_elimination()
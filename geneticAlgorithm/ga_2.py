# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/4 20:09
@Auth ： hujinghua
@File ：ga_2.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import numpy as np
import sklearn.svm
from sklearn.metrics import mean_squared_error
import math
from sklearn import ensemble

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    print("selected_elements_indices is :", selected_elements_indices)
    # print("features is :", features)
    reduced_features = features[:, selected_elements_indices]   #有问题，会出现index越界的情况
    return reduced_features

def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    print("correct is :", correct)
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def cal_pop_fitness(pop, features, train_X, test_X, train_y, test_y, fit_way, pop_1):
    accuracies = np.zeros(pop.shape[0])
    idx = 0
    # max_feature_reproducibility = 0;
    max_feature_reduction = 0;
    # min_feature_reproducibility = 9999;
    min_feature_reduction = 9999;
    feature_1 = []
    feature_2 = []
    for curr_solution in pop:
        pop_1.append(curr_solution)
        # reduced_features = reduce_features(curr_solution, features)
        train_data = reduce_features(curr_solution, train_X)
        test_data = reduce_features(curr_solution, test_X)
        # print("train_data is :", train_data.shape)
        # print("test_data is :", test_data.shape)
        train_labels = train_y
        test_labels = test_y.flatten()  # 把二维数组转换为一维数组
        if (fit_way == "classifier"):
            # 分类问题
            SV_classifier = sklearn.svm.SVC(gamma='scale')
            SV_classifier.fit(X=train_data, y=train_labels.astype('int'))
            predictions = SV_classifier.predict(test_data)
            print("predictions is :", predictions)
            print("labels is :", test_labels)
            accuracies[idx] = classification_accuracy(test_labels, predictions)
            idx = idx + 1

        # 回归问题
        elif (fit_way == "regressor"):
            # a = 1   #缩放系数
            m = features.shape[1]   #原始特征数
            m_1 = 0 # 该个体的特征数
            print("curr_solution is :", curr_solution)
            for feature in curr_solution:
                if (feature == 1):
                    m_1 = m_1 + 1
            if (m_1 == 0):
                continue
            randomRegressor = ensemble.RandomForestRegressor(n_estimators=20)
            randomRegressor.fit(X=train_data, y=train_labels)
            predictions = randomRegressor.predict(test_data)
            # print("predictions is :", predictions)
            # print("test_labels is :", test_labels)
            # feature_reduction表示降维率
            feature_reduction = 1 - (m - m_1) / m
            # print("feature_1 is :", feature_1)
            # feature_reduction = 1 / (1 + math.exp(-feature_1))
            feature_1.append((feature_reduction - 0.1) / (0.9 - 0.1))
            # print("降维率：", feature_reduction)

            # feature_reproducibility表示复现精度
            M_1 = 0
            for i in range(len(predictions)):
                y_predict = predictions[i]
                y_label = test_labels[i]
                M_1 = M_1 + math.sqrt((y_predict - y_label) ** 2)
            # feature_reproducibility = 1 / (1 + math.exp(-(-math.log10(M_1 / test_labels.shape[0]) - 6) / 2))
            feature_reproducibility = M_1 / test_labels.shape[0]
            feature_reproducibility = (feature_reproducibility - 0.03) / (0.13-0.03)
            # max_feature_reproducibility = max(max_feature_reproducibility, feature_reproducibility)
            # min_feature_reproducibility = min(min_feature_reproducibility, feature_reproducibility)
            print("复现精度 :", feature_reproducibility)
            feature_2.append(feature_reproducibility)
    for idx in range(len(feature_1)):
        try :
            accuracies[idx] = feature_1[idx] + feature_2[idx]
            if (accuracies[idx] == 0):
                print("异常值的时候", curr_solution)
        except:
            print("Exception")
    # max_feature_reproducibility = max(max_feature_reproducibility, 0)
    # min_feature_reproducibility = min(min_feature_reproducibility, 999)
    print("适应度 is :", accuracies)
    # print("max_feature_reproducibility :", max_feature_reproducibility)
    # print("min_feature_reproducibility :", min_feature_reproducibility)
    # print("max_feature_reduction :", max_feature_reduction)
    # print("min_feature_reduction :", min_feature_reduction)
    return accuracies

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=7):
    mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
         offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover
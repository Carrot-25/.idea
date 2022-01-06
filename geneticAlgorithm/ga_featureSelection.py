# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/4 20:48
@Auth ： hujinghua
@File ：ga_featureSelection.py
@IDE ：PyCharm
@Motto：基于遗传算法的特征约简
"""
import numpy as np
import ga_2 as GA
import pickle
import matplotlib.pyplot
import pandas as pd
from sklearn.model_selection import train_test_split

data_inputs  = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="E:N").values  #读取标准化后的数据表
data_outputs  = pd.read_excel(r'C:\Users\Carrot\Desktop\data.xlsx', usecols="O").values  #读取标准化后的数据表
num_samples = data_inputs.shape[0]
num_feature_elements = data_inputs.shape[1]
train_X, test_X, train_y, test_y = train_test_split(data_inputs, data_outputs, test_size=0.3, random_state = 5)
print("Number of training samples: ", train_X.shape[0])
print("Number of test samples: ", test_X.shape[0])

sol_per_pop = 100 # Population size
num_parents_mating = 4 # Number of parents inside the mating pool.
num_mutations = 4 # Number of elements to mutate.

# Defining the population shape.
pop_shape = (sol_per_pop, num_feature_elements)
# Creating the initial population.
new_population = np.random.randint(low=0, high=2, size=pop_shape)
print(new_population.shape)
best_outputs = []
num_generations = 1000
pop_1 = []

for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(new_population, data_inputs, train_X, test_X, train_y, test_y, fit_way = "regressor", pop_1 = pop_1)
    best_outputs.append(np.min(fitness))
    # The best result in the current iteration.
    print("Best result : ", best_outputs[-1])

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents, offspring_size=(pop_shape[0] - parents.shape[0], num_feature_elements))

    # Adding some variations to the offspring using mutation.
    offspring_mutation = GA.mutation(offspring_crossover, num_mutations=num_mutations)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation


# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(new_population, data_inputs, train_X, test_X, train_y, test_y, fit_way = "regressor", pop_1 = pop_1)

# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.min(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = new_population[best_match_idx, :]
best_solution_indices = np.where(best_solution == 1)[0]
best_solution_num_elements = best_solution_indices.shape[0]
best_solution_fitness = fitness[best_match_idx]

# print("pop_1 : ", pop_1)
print("best_match_idx : ", best_match_idx)
print("best_solution : ", best_solution)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_num_elements)
print("Best solution fitness : ", best_solution_fitness)

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
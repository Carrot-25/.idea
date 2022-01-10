# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 16:08
# @Author  : hujinghua
# @File    : ga.py
# @Software: IntelliJ IDEA
import pygad
from evolutionary_search import EvolutionaryAlgorithmSearchCV


function_inputs = [4,-2,3.5,5,-11,-4.7]  # Function inputs.
desired_output = 44  # Function output.

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

sol_per_pop = 50
num_genes = len(function_inputs)
init_range_low = -2
init_range_high = 5
mutation_percent_genes = 1

ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=20,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_percent_genes=mutation_percent_genes)
ga_instance.run()
ga_instance.plot_result()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
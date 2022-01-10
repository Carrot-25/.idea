# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 17:21
# @Author  : hujinghua
# @File    : ga_1.py
# @Software: IntelliJ IDEA
import torch
import pygad.torchga
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function
    """
    - model:模型
    - solution: 也就是遗传算法群体中的个体
    - data: 数据
    """
    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)
    # 计算误差
    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    # 因为评估值是越大越好
    solution_fitness = 1.0 / abs_error

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# 创建pytorch模型
input_layer = torch.nn.Linear(3 ,5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)

# 定义模型
model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

# 在初始化种群时，实例化 pygad.torchga.TorchGA
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=10)
# 定义 loss 函数
loss_function = torch.nn.L1Loss()

# 数据集输入数据
data_inputs = torch.tensor([[0.02, 0.1, 0.15],
                            [0.7, 0.6, 0.8],
                            [1.5, 1.2, 1.7],
                            [3.2, 2.9, 3.1]])

# 数据集
data_outputs = torch.tensor([[0.1],
                             [0.6],
                             [1.3],
                             [2.5]])

num_generations = 1000 # 迭代次数
num_parents_mating = 5 # 每次从父类中选择的个体进行交叉、和突变的数量
initial_population = torch_ga.population_weights # 初始化网络权重

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation
                       )

ga_instance.run()

ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# 返回最优参数的详细信息
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
# 基于最好的个体来进行预测
predictions = pygad.torchga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
print("Predictions : \n", predictions.detach().numpy())

abs_error = loss_function(predictions, data_outputs)
print("Absolute Error : ", abs_error.detach().numpy())
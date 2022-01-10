# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/3 16:36
@Auth ： hujinghua
@File ：rbfn.py
@IDE ：PyCharm
@Motto：径向基神经网络RBF
"""
import torch, random
import torch.nn as nn
import torch.optim as optim

class rbfn(nn.Module):
    def __init__(self, centers, n_out=10):
        self.n_out = n_out
        self.num_centers = centers.size(0)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        # self.linear = nn.Linear(self.num_centers + self.n_in, self.n_out, bias=True)
        self.linear = nn.Linear(self.num_centers, self.n_out, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
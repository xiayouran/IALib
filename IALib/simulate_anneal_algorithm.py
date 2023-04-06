# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/14 12:46
# Filename: simulate_anneal_algorithm.py
import numpy as np
import matplotlib.pyplot as plt

from .base_algorithm import BaseAlgorithm


__all__ = ['SimulateAnnealAlgorithm']


class SimulateAnnealAlgorithm(BaseAlgorithm):
    def __init__(self):
        super(SimulateAnnealAlgorithm, self).__init__()
        self.__max_count = 15     # 每个温度值得迭代次数
        self.__T = 100  # 初始温度
        self.__T_end = 1e-3  # 终止温度
        self.__coldrate = 0.9  # 冷却系数
        self.__x_range = (0, 5)
        self.__seed = 10086

        np.random.seed(self.__seed)

    def solution(self):
        x = np.random.uniform(*self.__x_range)  # 初始解
        while self.__T > self.__T_end:
            for _ in range(self.__max_count):
                y = self.problem_function(x)
                x_new = np.clip(x + np.random.randn(), a_min=self.__x_range[0], a_max=self.__x_range[1])
                y_new = self.problem_function(x_new)
                if y_new < y:  # 局部最优解
                    x = x_new
                else:
                    p = np.exp(-(y_new - y) / self.__T)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来接受最优解
                        x = x_new
            self.__T *= self.__coldrate

        print('({}, {})'.format(x, self.problem_function(x)))

    def draw(self):
        plt.figure()
        plt.ion()
        x_ = np.linspace(*self.__x_range, num=200)
        plt.plot(x_, self.problem_function(x_))

        x = np.random.uniform(*self.__x_range)  # 初始解
        while self.__T > self.__T_end:
            for _ in range(self.__max_count):
                y = self.problem_function(x)
                x_new = np.clip(x + np.random.randn(), a_min=self.__x_range[0], a_max=self.__x_range[1])

                # something about plotting
                if 'sca' in globals() or 'sca' in locals():
                    sca.remove()
                sca = plt.scatter(x, y, s=100, lw=0, c='red', alpha=0.5)
                plt.pause(0.01)

                y_new = self.problem_function(x_new)
                if y_new < y:  # 局部最优解
                    x = x_new
                else:
                    p = np.exp(-(y_new - y) / self.__T)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来跳出局部最优解
                        x = x_new
            self.__T *= self.__coldrate

        plt.scatter(x, self.problem_function(x), s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()
        print('({}, {})'.format(x, self.problem_function(x)))

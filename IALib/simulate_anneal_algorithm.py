# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/14 12:46
# Filename: simulate_anneal_algorithm.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .base_algorithm import BaseAlgorithm
from IALib.mixup.visu_func import Visu3DFunc


__all__ = ['SimulateAnnealAlgorithm']


class SimulateAnnealAlgorithm(BaseAlgorithm):
    def __init__(self, max_count=15, T=100, T_end=1e-3, coldrate=0.9, x_range=(0, 5), var_dim=1, seed=10086):
        super(SimulateAnnealAlgorithm, self).__init__()
        self.__max_count = max_count    # 每个温度值得迭代次数
        self.__T = T                    # 初始温度
        self.__T_end = T_end            # 终止温度
        self.__coldrate = coldrate      # 冷却系数
        self.__x_range = x_range        # 定义域
        self.__var_dim = var_dim        # 变量的个数
        self.__seed = seed
        self.optimal_solution = None

        np.random.seed(self.__seed)

    def problem_function(self, x):
        if self.__var_dim == 1:
            return super().problem_function(x)
        else:
            return Visu3DFunc.sphere(*x)

    def solution(self):
        x = np.random.uniform(*self.__x_range, size=self.__var_dim)  # 初始解
        while self.__T > self.__T_end:
            for _ in range(self.__max_count):
                y = self.problem_function(x)
                x_new = np.clip(x + np.random.randn(self.__var_dim), a_min=self.__x_range[0], a_max=self.__x_range[1])
                y_new = self.problem_function(x_new)
                if y_new < y:  # 局部最优解
                    x = x_new
                else:
                    p = np.exp(-(y_new - y) / self.__T)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来接受最优解
                        x = x_new
            self.__T *= self.__coldrate

        self.optimal_solution = (self.parse_format(x), self.parse_format(self.problem_function(x)))
        print('the optimal solution is', self.optimal_solution)

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

        self.optimal_solution = (self.parse_format(x), self.parse_format(self.problem_function(x)))
        print('the optimal solution is', self.optimal_solution)

    def draw3D(self):
        fig = plt.figure()
        plt.ion()
        ax = Axes3D(fig)
        x_ = np.linspace(*self.__x_range, num=200)
        X, Y = np.meshgrid(x_, x_)
        Z = self.problem_function([X, Y])
        ax.plot_surface(X, Y, Z, cmap=plt.cm.cool)
        ax.contour(X, Y, Z, levels=5, offset=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        x = np.random.uniform(*self.__x_range, size=self.__var_dim)  # 初始解
        while self.__T > self.__T_end:
            for _ in range(self.__max_count):
                y = self.problem_function(x)
                x_new = np.clip(x + np.random.randn(self.__var_dim), a_min=self.__x_range[0], a_max=self.__x_range[1])

                # something about plotting
                if 'sca' in globals() or 'sca' in locals():
                    sca.remove()
                sca = ax.scatter3D(*x, y, s=100, lw=0, c='red', alpha=0.5)
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

        ax.scatter3D(*x, self.problem_function(x), s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()

        self.optimal_solution = (self.parse_format(x), self.parse_format(self.problem_function(x)))
        print('the optimal solution is', self.optimal_solution)

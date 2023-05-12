# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/5/6 14:59
# Filename: ant_colony_optimization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .base_algorithm import BaseAlgorithm


__all__ = ['AntColonyOptimization']


class Ant:
    def __init__(self):
        self.id = None          # 蚂蚁所在区间的id
        self.eta = None         # 启发式信息, 存放本区间与其左右两个区间的函数差值
        self.tran_prob = None   # 状态转移概率


class Interval:
    def __init__(self):
        self.position = None    # 该区间上的中点
        self.tau = 1.           # 信息素浓度值
        self.delta_tau = 0.     # 信息素浓度增量
        self.count = 0          # 此区间上蚂蚁的数量


class AntColonyOptimization(BaseAlgorithm):
    def __init__(self, population_size=100, max_iter=200, alpha=1.5, beta=0.8, rho=0.3, epsilon=1e-4,
                 q=1.0, step=0.01, x_range=(0, 5), seed=10086):
        super(AntColonyOptimization, self).__init__()
        self.__population_size = population_size  # 蚂蚁种群大小
        self.__max_iter = max_iter  # 最大迭代次数
        self.__alpha = alpha        # 信息素重要程度因子
        self.__beta = beta          # 启发函数重要程度因子
        self.__rho = rho            # 信息素蒸发系数
        self.__epsilon = epsilon    # 停止搜索门限
        self.__q = q                # 信息素释放增量系数, 常量C
        self.__step = step          # 子区间间隔
        self.__x_range = x_range    # 变量x的定义域
        self.__population = []      # 蚁群
        self.__interval_set = []    # 区间集合(对应TSP问题中的city)
        self.__tabu = []            # 禁忌表
        self.__seed = seed
        self.optimal_solution = None

        np.random.seed(seed)

    def init_interval_set(self):
        for left_point in np.arange(*self.__x_range, self.__step):
            interval = Interval()
            interval.position = left_point + self.__step / 2  # 区间中点
            self.__interval_set.append(interval)

        if len(self.__interval_set) > self.__population_size:
            tmp_size = self.__population_size
            self.__population_size = int(np.ceil(len(self.__interval_set) / self.__population_size) * self.__population_size)
            print("Suggest a larger value for population_size, the value for population_size has been "
                  "changed from {} to {}".format(tmp_size, self.__population_size))

    def init_tabu(self):
        self.__tabu = np.zeros(shape=(len(self.__interval_set), 2))   # 初始禁忌表, 当前点与左右两坐标
        # TSP中的禁忌表的shape是(m, n), 其中m为蚂蚁数量, n为城市数量, 确保每只蚂蚁仅能访问城市一次

    def update_eta(self, ant):
        index = ant.id
        interval = self.__interval_set[index]
        ant.eta = []

        if index == 0:
            ant.eta.append(0)
            ant.eta.append(self.problem_function(interval.position) - self.problem_function(
                interval.position + self.__step))
        elif index == len(self.__interval_set) - 1:
            ant.eta.append(self.problem_function(interval.position) - self.problem_function(
                interval.position - self.__step))
            ant.eta.append(0)
        else:
            ant.eta.append(self.problem_function(interval.position) - self.problem_function(
                interval.position - self.__step))  # 当前区间(中点)与左邻居区间(中点)的差值
            ant.eta.append(self.problem_function(interval.position) - self.problem_function(
                interval.position + self.__step))  # 当前区间(中点)与右邻居区间(中点)的差值

    def update_tabu(self, ant):
        index = ant.id
        if ant.eta[0] > 0:
            self.__tabu[index, 0] = 1  # 表示左子区间值较小, 可以跳转
        if ant.eta[1] > 0:
            self.__tabu[index, 1] = 1  # 表示右子区间值较小, 可以跳转

    def init_population(self):
        # 初始化区间集合
        self.init_interval_set()

        # 初始化禁忌表
        self.init_tabu()

        for i in range(self.__population_size):
            index = np.random.choice(range(len(self.__interval_set)))   # 随机选择一个区间
            # index = i
            interval = self.__interval_set[index]
            interval.count += 1

            ant = Ant()
            ant.id = index

            # 更新eta
            self.update_eta(ant)

            # 更新禁忌表tabu
            self.update_tabu(ant)

            # 更新蚂蚁的状态转移概率
            ant.tran_prob = interval.tau * ant.eta[0] / (interval.tau * ant.eta[0] + interval.tau * ant.eta[1])  # 蚂蚁向左区间跳转概率

            self.__population.append(ant)

    def skip_left(self, ant):
        index = ant.id

        interval = self.__interval_set[index]
        interval.count -= 1  # 此区间蚂蚁数-1
        left_interval = self.__interval_set[index - 1]
        left_interval.count += 1  # 左区间蚂蚁数+1
        ant.id -= 1  # 蚂蚁跳转至左区间
        left_interval.delta_tau = left_interval.delta_tau + self.__q * ant.eta[0]

    def skip_right(self, ant):
        index = ant.id

        interval = self.__interval_set[index]
        interval.count -= 1  # 此区间蚂蚁数-1
        right_interval = self.__interval_set[index + 1]
        right_interval.count += 1  # 右区间蚂蚁数+1
        ant.id += 1  # 蚂蚁跳转至右区间
        right_interval.delta_tau = right_interval.delta_tau + self.__q * ant.eta[1]

    def search_local_optimal_solution(self):
        flag = np.ones(self.__population_size)
        while np.sum(flag):
            for i, ant in enumerate(self.__population):
                index = ant.id
                if self.__tabu[index, 0] and not self.__tabu[index, 1]:
                    # 蚂蚁可以向左区间跳转
                    self.skip_left(ant)
                elif not self.__tabu[index, 0] and self.__tabu[index, 1]:
                    # 蚂蚁可以向右区间跳转
                    self.skip_right(ant)
                elif self.__tabu[index, 0] and self.__tabu[index, 1]:
                    # 两个区间都可以跳转, 计算一下蚂蚁的状态转移概率
                    if ant.tran_prob > np.random.rand():
                        # 蚂蚁向左区间跳转
                        self.skip_left(ant)
                    else:
                        # 蚂蚁向右区间跳转
                        self.skip_right(ant)
                else:
                    flag[i] = 0     # 表示此蚂蚁不再进行跳转了

                self.update_eta(ant)    # 更新eta
                self.update_tabu(ant)   # 更新禁忌表

            for interval in self.__interval_set:
                # 更新区间上的信息素
                interval.tau = (1 - self.__rho) * interval.tau + interval.delta_tau

    def print_local_optimal_solution(self):
        print('local optimal solution:')
        local_optimal_solution = {}
        best_point = np.inf
        for ant in self.__population:
            index = ant.id
            if not local_optimal_solution.get(index, ''):
                local_optimal_solution[index] = (self.__interval_set[index].position,
                                                 self.problem_function(self.__interval_set[index].position))
                print(local_optimal_solution[index])
                if best_point > local_optimal_solution[index][1]:
                    best_point = local_optimal_solution[index][1]
                    self.optimal_solution = local_optimal_solution[index]

    def solution(self):
        self.init_population()
        self.search_local_optimal_solution()
        self.print_local_optimal_solution()

        print('the optimal solution is', self.optimal_solution)


    def draw(self):
        self.init_population()

        plt.figure()
        plt.ion()
        x = np.linspace(*self.__x_range, 200)
        plt.plot(x, self.problem_function(x))

        def collect_points():
            x_tmp = []
            for ant in self.__population:
                x_tmp.append(self.__interval_set[ant.id].position)
            sca = plt.scatter(np.asarray(x_tmp), self.problem_function(np.asarray(x_tmp)), s=100, lw=0, c='red',
                              alpha=0.5)
            plt.pause(0.05)

            return sca

        sca = collect_points()
        flag = np.ones(self.__population_size)
        while np.sum(flag):
            for i, ant in enumerate(self.__population):
                index = ant.id
                if self.__tabu[index, 0] and not self.__tabu[index, 1]:
                    # 蚂蚁可以向左区间跳转
                    self.skip_left(ant)
                elif not self.__tabu[index, 0] and self.__tabu[index, 1]:
                    # 蚂蚁可以向右区间跳转
                    self.skip_right(ant)
                elif self.__tabu[index, 0] and self.__tabu[index, 1]:
                    # 两个区间都可以跳转, 计算一下蚂蚁的状态转移概率
                    if ant.tran_prob > np.random.rand():
                        # 蚂蚁向左区间跳转
                        self.skip_left(ant)
                    else:
                        # 蚂蚁向右区间跳转
                        self.skip_right(ant)
                else:
                    flag[i] = 0     # 表示此蚂蚁不再进行跳转了

                self.update_eta(ant)
                self.update_tabu(ant)

            for interval in self.__interval_set:
                # 更新区间上的信息素
                interval.tau = (1 - self.__rho) * interval.tau + interval.delta_tau

            if 'sca' in globals() or 'sca' in locals():
                sca.remove()
            sca = collect_points()

        plt.ioff()
        plt.show()

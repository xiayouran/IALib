# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/30 15:50
# Filename: pso_saa.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IALib.base_algorithm import BaseAlgorithm
from IALib.particle_swarm_optimization import ParticleSwarmOptimization, Particle
from IALib.simulate_anneal_algorithm import SimulateAnnealAlgorithm
from IALib.mixup.visu_func import Visu3DFunc


__all__ = ['PSO_SAA']


class PSO_SAA(BaseAlgorithm):
    def __init__(self, population_size=100, p_dim=1, v_dim=1, max_iter=500, x_range=(0, 5),
                 t_max=1.0, t_min=1e-3, coldrate=0.9, seed=10086):
        super(PSO_SAA, self).__init__()
        self.__population_size = population_size  # 种群大小
        self.__p_dim = p_dim        # 粒子位置维度
        self.__v_dim = v_dim        # 粒子速度维度
        self.__max_iter = max_iter  # 最大迭代次数
        self.__t_max = t_max  # 初始温度
        self.__t_min = t_min  # 终止温度
        self.__coldrate = coldrate  # 降温速率
        self.saa_best_particle = None   # 模拟退火算法得到的最优解
        self.best_particle = None       # 最优解
        self.__x_range = x_range
        self.__seed = seed
        self.optimal_solution = None

        np.random.seed(self.__seed)

    def problem_function(self, x):
        if self.__p_dim == 1:
            return super().problem_function(x)
        else:
            return Visu3DFunc.sphere(*x)

    def solution(self):
        # PSO
        algo_pso = ParticleSwarmOptimization(population_size=self.__population_size,
                                             p_dim=self.__p_dim, v_dim=self.__v_dim,
                                             max_iter=self.__max_iter, x_range=self.__x_range)
        algo_pso.solution()

        # SAA
        x = algo_pso.global_best_particle.best_position   # 初始解
        while self.__t_max > self.__t_min:
            for _ in range(self.__max_iter):
                x_new = np.clip(x + np.random.randn(), a_min=self.__x_range[0], a_max=self.__x_range[1])
                delta = self.problem_function(x_new) - self.problem_function(x)  # 计算目标函数的值差
                if delta < 0:  # 局部最优解
                    x = x_new   # 直接接受更优解
                else:
                    p = np.exp(-delta / self.__t_max)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来接受最优解
                        x = x_new
            self.__t_max *= self.__coldrate

        # optimal solution
        saa_best_particle = Particle()
        saa_best_particle.position = x
        saa_best_particle.best_position = x
        saa_best_particle.fitness = self.problem_function(x)
        self.saa_best_particle = saa_best_particle

        if saa_best_particle.fitness < algo_pso.global_best_particle.fitness:
            self.best_particle = saa_best_particle
        else:
            self.best_particle = algo_pso.global_best_particle

        self.optimal_solution = (self.parse_format(self.best_particle.position),
                                 self.parse_format(self.best_particle.fitness))
        print('the optimal solution is', self.optimal_solution)
        # print('optimal solution:\nposition: {} \nfitness: {}'.format(self.best_particle.best_position,
        #                                                              self.best_particle.fitness))

    def draw(self):
        # PSO
        algo_pso = ParticleSwarmOptimization(population_size=self.__population_size,
                                             p_dim=self.__p_dim, v_dim=self.__v_dim,
                                             max_iter=self.__max_iter, x_range=self.__x_range)
        algo_pso.draw(mixup=True)
        plt.clf()
        x = np.linspace(*self.__x_range, 200)
        plt.plot(x, self.problem_function(x))

        # SAA
        x = algo_pso.global_best_particle.best_position   # 初始解
        while self.__t_max > self.__t_min:
            for _ in range(self.__max_iter):
                # something about plotting
                if 'sca' in globals() or 'sca' in locals():
                    sca.remove()
                sca = plt.scatter(x, self.problem_function(x), s=100, lw=0, c='red', alpha=0.5)
                plt.pause(0.01)

                x_new = np.clip(x + np.random.randn(), a_min=self.__x_range[0], a_max=self.__x_range[1])
                delta = self.problem_function(x_new) - self.problem_function(x)  # 计算目标函数的值差
                if delta < 0:  # 局部最优解
                    x = x_new   # 直接接受更优解
                else:
                    p = np.exp(-delta / self.__t_max)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来接受最优解
                        x = x_new
            self.__t_max *= self.__coldrate

        # optimal solution
        saa_best_particle = Particle()
        saa_best_particle.position = x
        saa_best_particle.best_position = x
        saa_best_particle.fitness = self.problem_function(x)
        self.saa_best_particle = saa_best_particle

        if saa_best_particle.fitness < algo_pso.global_best_particle.fitness:
            self.best_particle = saa_best_particle
        else:
            self.best_particle = algo_pso.global_best_particle

        plt.scatter(self.best_particle.best_position, self.best_particle.fitness, s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()

        self.optimal_solution = (self.parse_format(self.best_particle.position),
                                 self.parse_format(self.best_particle.fitness))
        print('the optimal solution is', self.optimal_solution)
        # print('optimal solution:\nposition: {} \nfitness: {}'.format(self.best_particle.best_position,
        #                                                              self.best_particle.fitness))

    def draw3D(self):
        # PSO
        algo_pso = ParticleSwarmOptimization(population_size=self.__population_size,
                                             p_dim=self.__p_dim, v_dim=self.__v_dim,
                                             max_iter=self.__max_iter, x_range=self.__x_range)
        algo_pso.draw3D(mixup=True)
        plt.clf()
        ax = Axes3D(algo_pso.fig)
        x_ = np.linspace(*self.__x_range, num=200)
        X, Y = np.meshgrid(x_, x_)
        Z = self.problem_function([X, Y])
        ax.plot_surface(X, Y, Z, cmap=plt.cm.cool)
        ax.contour(X, Y, Z, levels=5, offset=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # SAA
        x = algo_pso.global_best_particle.best_position   # 初始解
        while self.__t_max > self.__t_min:
            for _ in range(self.__max_iter):
                # something about plotting
                if 'sca' in globals() or 'sca' in locals():
                    sca.remove()
                sca = ax.scatter3D(*x, self.problem_function(x), s=100, lw=0, c='red', alpha=0.5)
                plt.pause(0.01)

                x_new = np.clip(x + np.random.randn(), a_min=self.__x_range[0], a_max=self.__x_range[1])
                delta = self.problem_function(x_new) - self.problem_function(x)  # 计算目标函数的值差
                if delta < 0:  # 局部最优解
                    x = x_new   # 直接接受更优解
                else:
                    p = np.exp(-delta / self.__t_max)  # 粒子在温度T时趋于平衡的概率为exp[-ΔE/(kT)]
                    r = np.random.uniform(0, 1)
                    if p > r:  # 以一定概率来接受最优解
                        x = x_new
            self.__t_max *= self.__coldrate

        # optimal solution
        saa_best_particle = Particle()
        saa_best_particle.position = x
        saa_best_particle.best_position = x
        saa_best_particle.fitness = self.problem_function(x)
        self.saa_best_particle = saa_best_particle

        if saa_best_particle.fitness < algo_pso.global_best_particle.fitness:
            self.best_particle = saa_best_particle
        else:
            self.best_particle = algo_pso.global_best_particle

        ax.scatter3D(*self.best_particle.best_position, self.best_particle.fitness, s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()

        self.optimal_solution = (self.parse_format(self.best_particle.position),
                                 self.parse_format(self.best_particle.fitness))
        print('the optimal solution is', self.optimal_solution)
        # print('optimal solution:\nposition: {} \nfitness: {}'.format(self.best_particle.best_position,
        #                                                              self.best_particle.fitness))


if __name__ == '__main__':
    algo = PSO_SAA()
    # algo.draw()
    algo.draw3D()

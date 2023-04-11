# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/16 16:54
# Filename: particle_swarm_optimization.py
import numpy as np
import matplotlib.pyplot as plt
from .base_algorithm import BaseAlgorithm


__all__ = ['ParticleSwarmOptimization']


class Particle:
    def __init__(self):
        self.position = None    # 粒子的位置
        self.velocity = None    # 粒子的速度
        self.best_position = None   # 个体最优解
        self.fitness = None         # 适应度值


class ParticleSwarmOptimization(BaseAlgorithm):
    def __init__(self, population_size=100, p_dim=1, v_dim=1, max_iter=500, x_range=(0, 5), seed=10086):
        super(ParticleSwarmOptimization, self).__init__()
        self.__population_size = population_size  # 种群大小
        self.__p_dim = p_dim        # 粒子位置维度
        self.__v_dim = v_dim        # 粒子速度维度
        self.__max_iter = max_iter  # 最大迭代次数
        self.__w = 0.5    # 惯性权重
        self.__c1 = 1.5   # 加速因子1
        self.__c2 = 1.5   # 加速因子2
        self.__population = []    # 粒子群
        self.global_best_particle = None    # 全局最优解
        self.__x_range = x_range
        self.__seed = seed

        np.random.seed(seed)

    def init_population(self):
        for i in range(self.__population_size):
            particle = Particle()
            particle.position = np.random.uniform(*self.__x_range, size=self.__p_dim)   # 随机初始化位置
            particle.velocity = np.random.uniform(-1, 1, size=self.__v_dim)     # 随机初始化速度
            particle.best_position = particle.position  # 初始最优位置
            particle.fitness = self.problem_function(particle.position)     # 计算适应度值
            if self.global_best_particle is None or particle.fitness < self.problem_function(self.global_best_particle.position):
                self.global_best_particle = particle  # 更新全局最优解
            self.__population.append(particle)

    def update_population(self):
        for i in range(self.__population_size):
            particle = self.__population[i]
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            # Step2: 更新速度
            particle.velocity = self.__w * particle.velocity + \
                                self.__c1 * r1 * (particle.best_position - particle.position) + \
                                self.__c2 * r2 * (self.global_best_particle.position - particle.position)
            # Step3: 更新位置
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, 0, 5)
            # Step4: 更新个体最优解
            if self.problem_function(particle.position) < self.problem_function(particle.best_position):
                particle.best_position = particle.position
                particle.fitness = self.problem_function(particle.position)
            # Step5: 更新全局最优解
            if self.problem_function(particle.position) < self.problem_function(self.global_best_particle.position):
                self.global_best_particle = particle

    def solution(self):
        # Step1: 粒子初始化
        self.init_population()
        for i in range(self.__max_iter):
            self.update_population()
        print('best particle:\nposition: {}\nvelocity: {}'
              '\nfitness: {}\nbest_position: {}'.format(self.global_best_particle.position,
                                                        self.global_best_particle.velocity,
                                                        self.global_best_particle.fitness,
                                                        self.global_best_particle.best_position))

    def draw(self):
        self.init_population()

        plt.figure()
        plt.ion()
        x = np.linspace(*self.__x_range, 200)
        plt.plot(x, self.problem_function(x))

        for i in range(self.__max_iter):
            particle_xy = [[particle.position, particle.fitness] for particle in self.__population]
            particle_xy = np.asarray(particle_xy)
            # something about plotting
            if 'sca' in globals() or 'sca' in locals():
                sca.remove()
            sca = plt.scatter(particle_xy[:, 0], particle_xy[:, 1], s=100, lw=0, c='red', alpha=0.5)
            plt.pause(0.05)

            self.update_population()

        plt.scatter(self.global_best_particle.position, self.global_best_particle.fitness, s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()
        print('({}, {})'.format(self.global_best_particle.position, self.global_best_particle.fitness))

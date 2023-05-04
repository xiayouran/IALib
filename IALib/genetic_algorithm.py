# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/13 20:58
# Filename: genetic_algorithm.py
# CSDNblog: https://blog.csdn.net/qq_42730750/article/details/129444149
import numpy as np
import matplotlib.pyplot as plt

from .base_algorithm import BaseAlgorithm


__all__ = ['GeneticAlgorithm']


class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, dna_size=10,
                 population_size=100,
                 crossover_rate=0.7,
                 mutation_rate=0.003,
                 max_generations=1000,
                 x_range=(0, 5),
                 seed=10086):
        super(GeneticAlgorithm, self).__init__()
        self.__dna_size = dna_size                  # DNA length
        self.__population_size = population_size    # population size
        self.__crossover_rate = crossover_rate      # mating probability(DNA crossover)
        self.__mutation_rate = mutation_rate        # mutation probability
        self.__max_generations = max_generations    # maximum iterations
        self.__x_range = x_range                    # x upper and lower bounds
        self.__seed = seed                          # random seed
        self.optimal_solution = None

        np.random.seed(self.__seed)

    def __get_fitness(self, pred):
        """find non-zero fitness for selection"""
        return pred + 1e-3 - np.min(pred)

    def __translateDNA(self, population):
        """convert binary DNA to decimal and normalize it to a range"""
        return population.dot(2 ** np.arange(self.__dna_size)[::-1]) / float(2 ** self.__dna_size - 1) * self.__x_range[1]

    def __selection(self, population, fitness):
        # p: 一维数组, 决定了数组中每个元素采样的概率, 默认为None, 即每个元素被采样的概率相同
        # replace=True, 允许元素重复
        idx = np.random.choice(np.arange(self.__population_size), size=self.__population_size, replace=True,
                               p=fitness / fitness.sum())

        return population[idx]

    def __crossover(self, parent, population):
        if np.random.rand() < self.__crossover_rate:  # random crossover
            i_ = np.random.randint(0, self.__population_size, size=1)  # select another individual from population
            cross_points = np.random.randint(0, 2, size=self.__dna_size).astype(np.bool)  # choose crossover points
            parent[cross_points] = population[i_, cross_points]  # mating and produce one child

        return parent

    def __mutation(self, child):
        for point in range(self.__dna_size):
            if np.random.rand() < self.__mutation_rate:  # random mutate
                child[point] = 1 if child[point] == 0 else 0

        return child

    def solution(self):
        # Step1: initialize the population DNA
        population = np.random.randint(2, size=(self.__population_size, self.__dna_size))

        for _ in range(self.__max_generations):
            f_values = self.problem_function(self.__translateDNA(population))

            # Step2: compute fitness value
            fitness = self.__get_fitness(f_values)
            best_id = np.argmax(fitness)
            self.optimal_solution = (self.parse_format(self.__translateDNA(population[best_id])),
                                     self.parse_format(self.problem_function(self.__translateDNA(population[best_id]))))
            # print("Most fitted DNA: {}, x: {}, max_value: {}".format(population[best_id],
            #                                                          self.__translateDNA(population[best_id]),
            #                                                          self.problem_function(
            #                                                              self.__translateDNA(population[best_id]))))
            # Step3: selection
            population = self.__selection(population, fitness)
            population_copy = population.copy()
            for parent in population:
                # Step4: crossover
                child = self.__crossover(parent, population_copy)
                # Step5: mutation
                child = self.__mutation(child)
                parent[:] = child  # parent is replaced by its child

        print('the optimal solution is', self.optimal_solution)

    def draw(self):
        # Step1: initialize the population DNA
        population = np.random.randint(2, size=(self.__population_size, self.__dna_size))

        plt.figure()
        plt.ion()
        x = np.linspace(*self.__x_range, 200)
        plt.plot(x, self.problem_function(x))

        for _ in range(self.__max_generations):
            f_values = self.problem_function(self.__translateDNA(population))

            # something about plotting
            if 'sca' in globals() or 'sca' in locals():
                sca.remove()
            sca = plt.scatter(self.__translateDNA(population), f_values, s=100, lw=0, c='red', alpha=0.5)
            plt.pause(0.05)

            # Step2: compute fitness value
            fitness = self.__get_fitness(f_values)
            best_id = np.argmax(fitness)
            self.optimal_solution = (self.parse_format(self.__translateDNA(population[best_id])),
                                     self.parse_format(self.problem_function(self.__translateDNA(population[best_id]))))
            # print("Most fitted DNA: {}, x: {}, max_value: {}".format(population[best_id],
            #                                                          self.__translateDNA(population[best_id]),
            #                                                          self.problem_function(
            #                                                              self.__translateDNA(population[best_id]))))
            # Step3: selection
            population = self.__selection(population, fitness)
            population_copy = population.copy()
            for parent in population:
                # Step4: crossover
                child = self.__crossover(parent, population_copy)
                # Step5: mutation
                child = self.__mutation(child)
                parent[:] = child  # parent is replaced by its child

        plt.scatter(self.optimal_solution[0], self.optimal_solution[1], s=100, lw=0, c='green', alpha=0.7)
        plt.ioff()
        plt.show()

        print('the optimal solution is', self.optimal_solution)

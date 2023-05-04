# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/14 10:13
# Filename: base_algorithm.py
import numpy as np


__all__ = ['BaseAlgorithm']


class BaseAlgorithm(object):
    """Base class of algorithm"""

    def __init__(self):
        pass

    def problem_function(self, x):
        """Expression of the problem"""
        # f(x) = xsin(5x) - xcos(2x)
        return x*np.sin(5*x) - x*np.cos(2*x)

    def solution(self):
        """Solution to the problem"""
        pass

    def draw(self):
        """Visualize the problem solving process"""
        pass

    def parse_format(self, x):
        """Unified data format"""
        if isinstance(x, np.ndarray):
            if x.size == 1:
                x = x[0]
            elif x.size == 2:
                x = x.tolist()
        else:
            pass

        return x

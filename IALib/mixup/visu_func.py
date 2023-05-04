# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/30 14:22
# Filename: visu_func.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visu3DFunc(object):
    def __init__(self, func_name='Sphere'):
        self.func_name = func_name
        self.X = np.linspace(-5, 5, num=200)
        self.Y = np.linspace(-5, 5, num=200)

    @classmethod
    def sphere(cls, x, y):
        """Sphere"""
        return x**2 + y**2

    @classmethod
    def himmelblau(cls, x, y):
        """Himmelblau"""
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    @classmethod
    def ackley(cls, x, y, a=20, b=0.2, c=2*np.pi):
        """Ackley"""
        term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2)/2))
        term2 = -np.exp((np.cos(c*x) + np.cos(c*y))/2)
        return term1 + term2 + a + np.exp(1)

    def draw(self):
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        X, Y = np.meshgrid(self.X, self.Y)

        if self.func_name == 'Sphere':
            Z = self.sphere(X, Y)
        elif self.func_name == 'Himmelblau':
            Z = self.himmelblau(X, Y)
        else:
            Z = self.ackley(X, Y)

        ax.plot_surface(X, Y, Z, cmap=plt.cm.cool)
        ax.contour(X, Y, Z, levels=5, offset=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('{} Function'.format(self.func_name))
        # ax.scatter3D(0, 0, self.sphere(0, 0), s=100, lw=0, c='green', alpha=0.7)
        plt.savefig(self.func_name)

        plt.show()


if __name__ == '__main__':
    # Sphere, Himmelblau, Ackley
    visu_obj = Visu3DFunc(func_name='Sphere')
    visu_obj.draw()


# -*- coding:utf-8 -*-
# Author:   xiayouran
# Email:    youran.xia@foxmail.com
# Datetime: 2023/3/14 11:17
# Filename: main.py
import sys
from pathlib import Path

lib_dir = Path(__file__).resolve().parent.parent
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import IALib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-algo', type=str, default='pso', choices=['ga', 'saa', 'pso'],
                    help='select a algorithm')
args = parser.parse_args()


if __name__ == '__main__':
    algo_name = args.algorithm

    if algo_name == 'ga':
        algo = IALib.GeneticAlgorithm()
        algo.solution()
    elif algo_name == 'saa':
        algo = IALib.SimulateAnnealAlgorithm()
        algo.solution()
    elif algo_name == 'pso':
        algo = IALib.ParticleSwarmOptimization()
        algo.solution()

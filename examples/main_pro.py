# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2023/4/15 20:46
# Filename: main_pro.py
import sys
from pathlib import Path

lib_dir = Path(__file__).resolve().parent.parent
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import IALib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-algo', type=str, default='pso_saa', choices=['saa', 'pso', 'pso_saa'],
                    help='select a algorithm')
args = parser.parse_args()


if __name__ == '__main__':
    algo_name = args.algorithm

    if algo_name == 'saa':
        algo = IALib.SimulateAnnealAlgorithm(max_count=30, T=100, T_end=1e-5, coldrate=0.95, x_range=(-5, 5), var_dim=2)
        algo.solution()     # draw3D()
    elif algo_name == 'pso':
        algo = IALib.ParticleSwarmOptimization(p_dim=2, v_dim=2, max_iter=20, x_range=(-5, 5))
        algo.solution()     # draw3D()
    elif algo_name == 'pso_saa':
        algo = IALib.PSO_SAA(p_dim=2, v_dim=2, max_iter=20, x_range=(-5, 5))
        algo.solution()     # draw3D()

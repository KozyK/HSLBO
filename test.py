# -*- coding: utf-8 -*-

import numpy as np
import BO

if __name__=='__main__':
    task = BO.SimpleTask()
    bo = BO.BayesianOptimization(task)
    bo.sequential_optimize(1)
    print(bo.inputs)
    print(bo.values)
    print("Current best is {0}".format(bo.current_best))

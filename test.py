# -*- coding: utf-8 -*-

import numpy as np
import hslbo
import matplotlib.pyplot as plt

from HSLBO.example.branin import branin, branin_modified

def sequential_plot(bo, itertimes):
    MAXIMUM_BRANIN = 0.397887

    num =[]
    optimum = []

    best = np.squeeze(bo.current_best[1])
    optimum.append(best)
    num.append(0)

    for i in range(itertimes):
        bo.update()
        best = np.squeeze(bo.current_best[1])
        optimum.append(best)
        num.append(i+1)

    plt.plot(num,optimum)
    plt.xlabel("Number of iteration")
    plt.ylabel("Optimum value")
    plt.title("BO GP-MI")
    plt.show()


if __name__=='__main__':
    func = branin
    num_dims = 2
    lower = np.array([-5,0]) # 下界
    upper = np.array([10,15]) # 上界

    bo = HSLBO.BO(func, num_dims, lower, upper, acq_name="mutual_information")

    #bo.sequential_update(150)
    sequential_plot(bo, 300)

    inputs = bo.inputs
    outputs = bo.values
    current_best = bo.current_best

    print("Inputs:\n{0}".format(inputs))
    print("Outputs:\n{0}".format(outputs))
    print("Current best:\n{0}".format(current_best[0]))
    print("Current best:\n{0}".format(current_best[1]))

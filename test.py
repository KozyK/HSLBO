# -*- coding: utf-8 -*-

import numpy as np
import hslbo
import matplotlib.pyplot as plt

from hslbo.example.branin import branin, branin_modified


# 最適解の推移をプロットする
# TODO: 内部で最適化するのではなく，探索終了後にデータを与えることでプロットする
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

    func = branin_modified

    num_dims = 2

    # 最初の一点を除く探索回数
    num_iter = 10

    lower = np.array([-5,0]) # 下界
    upper = np.array([10,15]) # 上界

    acq_options = {"mcmc_iters": 300, "burnin": 3000}

    bo = hslbo.BO(func, num_dims, lower, upper, acq_name="mutual_information", acq_options=acq_options)

    bo.sequential_update(num_iter)
    #sequential_plot(bo, 300)

    inputs = bo.inputs
    outputs = bo.values
    current_best = bo.current_best

    print("Inputs:\n{0}".format(inputs))
    print("Outputs:\n{0}".format(outputs))
    print("Current optimal point:\n{0}".format(current_best[0]))
    print("Current optimum:\n{0}".format(current_best[1]))

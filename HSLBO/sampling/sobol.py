# -*- coding: utf-8 -*-
# sobol.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

#import time
import numpy as np
import sobol_seq
#import scipy.stats as sps
#import matplotlib.pyplot as plt
#from pyDOE import lhs


def sobol_sequence(dim, num, loc=0, scale=1):
    """ A function produces Sobol sequences.

    Parameters
    ----------
    dim: integer
        number of dimension
    num: integer
        number of samples
    loc: array, float
        lower bound (,dim)
    scale: array, float
        scale length of the hypercube (,dim)


    NOTE
    ----
    乱数列を使わない決定論的な準モンテカルロ法なので，サンプル数を固定すれば毎回呼び出す必要はない？

        If you know in advance the number n of points that you plan to generate, some authors suggest that better uniformity can be attained by first skipping the initial portion of the LDS (and in particular, the first power of two smaller than n; see Joe and Kuo, 2003)

    つまり，numより小さい最大の2^m をスキップすると高い一様性が得られるらしい．ソースによってはnumより大きい最小の2^mって書いてる・・・どっちやねん．“目で”検証した結果，少ないサンプルでは後者の方が一様なサンプルが出来ていたので後者を採用することにした．
    """
    skip = 1
    while skip <= num:
        skip = skip*2
    sobol = sobol_seq.i4_sobol_generate(dim, num+skip)[skip:,:]
    # 次元のシャッフル
    sobol = sobol[:,np.random.permutation(np.arange(dim))]
    return sobol*scale + loc


def plot_scatter(results):
    num = len(results) # 手法の数
    axs = []
    fig = plt.figure()

    for i, (v, k)  in enumerate(results):
        ax = fig.add_subplot(((num+2)//3)*100+30+i+1) # 3列のサブプロット
        h = ax.scatter(v[:,0], v[:,1], facecolor='None', edgecolor='blue', marker='o', alpha=50)
        # N=1-10, 11-100 101-256ごとに色を変えて散布図をプロット
#        h1 = ax.scatter(v[:10,0], v[:10,1], facecolor='None', edgecolor='red', marker='o')
#        h2 = ax.scatter(v[10:100,0], v[10:100,1], facecolor='None', edgecolor='blue', marker='o')
#        h3 = ax.scatter(v[100:,0], v[100:,1], facecolor='None', edgecolor='green', marker='o')
        ax.set_aspect('1.0') # サブプロットのアスペクト比
        ax.axis([0, 1, 0, 1]) # スケール
        ax.set_xlabel(k) # ラベル
        ax.tick_params(axis='both', which='both', bottom='off', top='off',right='off',left='off', labelbottom='off',labelleft='off')
        axs.append(ax)

#    fig.legend(handles=(h1), labels=("N=1-10", "N=11-100" ,"N=101-256"), loc=(0.85,0.80),fontsize=10)
    fig.subplots_adjust(left=0.02, bottom=0.05, right=0.80, top=0.95, wspace=0.05, hspace=0.1)
    plt.suptitle("Uniform sampling of unit hypercube") # 表全体のサブタイトルをつける
    plt.show()

if __name__ == '__main__':
    """
    num=1000
    dim=2
    start = time.time()
    for i in range(100):
        s = sobol_sequence(2,1000)
    elapsed_time = (time.time() - start)/100
    print(s.shape)
    print("elapsed_time:{0}".format(elapsed_time))

    num_samples = 256

    # Monte Carlo (Uniform distribution)
    mc = sps.uniform(loc=0,scale=1).rvs(size=(num_samples,2))
    # Latin Hypercube
    latin0 = lhs(2, samples=num_samples)
    # Sobol sequence
    sobol = sobol_seq.i4_sobol_generate(2,num_samples)
    # Latin Hypercube(center)
    latin1 = lhs(2, samples=num_samples, criterion="center")
    # Latin Hypercube(maximized minimum distance)
    latin2 = lhs(2, samples=num_samples, criterion="maximin")
    # Latin Hypercube(minimized maximum correlation)
    latin3 = lhs(2, samples=num_samples, criterion="corr")

    results = [(mc, "Monte Carlo"),
               (latin0, "Latin hypercube"),
               (sobol, "Sobol sequence"),
               (latin1, "Latin hypercube (center)"),
               (latin2, "Latin hypercube (maxmindist)"),
               (latin3, "Latin hypercube (minmaxcorr)")]
    """

    # Sobol sequence
    sobol0 = sobol_sequence(5,100)[:,0:2]
    # Sobol sequence
    sobol1 = sobol_sequence(5,100)[:,1:3]
    # Sobol sequence
    sobol2 = sobol_sequence(5,100)[:,2:4]
    # Sobol sequence
    sobol3 = sobol_seq.i4_sobol_generate(5,100)[:,0:2]
    # Sobol sequence
    sobol4 = sobol_seq.i4_sobol_generate(5,100)[:,1:3]
    # Sobol sequence
    sobol5 = sobol_seq.i4_sobol_generate(5,100)[:,2:4]

    results_s = [(sobol0, "Sobol with skipping"),
           (sobol1, "Sobol with skipping"),
           (sobol2, "Sobol with skipping"),
           (sobol3, "Sobol 100"),
           (sobol4, "Sobol 200"),
           (sobol5, "Sobol 300")]

    plot_scatter(results_s)

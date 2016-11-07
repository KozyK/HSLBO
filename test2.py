# -*- coding: utf-8 -*-

if __name__=='__main__':
    import os

    try:
        import cPickle as pickle
    except:
        import pickle

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np


    # 結果の格納されたオブジェクトを読み込み
    base = os.path.dirname(__file__)
    name = os.path.join(base, 'result/result01_GPMI_noiseless.pickle')
    with open(name, mode='rb') as f:
        results = pickle.load(f)

    # 結果をプロットする

    # 探索点の数を取得
    num_iter = results[0]["opt_f_hist"].size

    # 最適化の履歴を縦に並べる
    hist = np.array([result["opt_f_hist"] for result in results])

    # 平均と標準偏差をとる
    mean = np.mean(hist,axis=0)
    sigma = np.std(hist, axis=0, ddof=1)

    # 平均と標準偏差をプロット
    x = np.arange(num_iter) + 1
    plt.plot(x, mean, color='b')
    plt.fill_between(x, mean-sigma, mean+sigma, alpha=0.5, color='b')
    # 最適値を赤い水平線でプロット
    MINIMUM_BRANIN_MOD = -15.31007
    plt.axhline(y=MINIMUM_BRANIN_MOD, color='red')

    plt.legend(('GP-MI',))
    plt.xlim(0, num_iter)
    plt.xlabel("Number of iteration")
    plt.ylabel("Minimimum function value")
    plt.title("Modified Branin function")

    plt.savefig('result01_figure.pdf')
    plt.show()

# -*- coding: utf-8 -*-
# multistart_minimizer.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

import numpy as np
import scipy.optimize as spo

# pyDOEを別途インストールしてください
from pyDOE import lhs

# TODO: 可変長引数optim_optionsの渡し方（制約条件とか）
# TODO: scipy.optimize.OptimizeResultが勝手にndarrayに変換されてしまう不具合．scipy.optimize.minimizeの代わりにscipy.optimize.l_bfgs_bを用いることで回避しています

def multistart_minimizer(fun, lower, upper, init_guesses=None, num_init=None, **optim_options):
    """ 複数の初期値を用いて勾配を用いた非線形最適化を実行する


    Parameters
    ----------
    fun : callable
        最小化したい評価関数
    lower : array [num_dims,]
        探索空間の下界
    upper : array [num_dims,]
        探索空間の上界
    init_guesses : array-like, (num_samples, num_dims)
        初期値（解の候補）を格納した2次元配列
    num_init : integer
        初期値の数

    未実装箇所
    **optim_option : dict
        scipy.optimize.minimizeに通す
        その他のオプション(constraints, method とか)
        参照 :
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Returns
    -------
    result : Tuple
        最も良かった最適化の結果を返す.
            result[0] : 最適解
            result[1] : 最適解における関数値
    """

    # もし初期値も初期値の数も与えられなければエラー
    if (init_guesses is None) and (num_init is None):
        raise Exception("Initial guesses or the number of initial guess must be specified.")
    # 両方与えられれば初期値の方を優先
    elif (not init_guesses is None) and (not num_init is None):
        print("Multistart: Assigned initial guesses are used for optimization")
    # 初期値がない場合はラテン方格で作成
    elif (init_guesses is None) and (not num_init is None):
        init_guesses = lhs(lower.size, samples=num_init, criterion='center') * (upper-lower) + lower

    # 1次元配列(初期値が1つだけ)なら2次元配列に変更しておく(やり方汚いかも)
    if init_guesses.shape[0] == lower.size:
        init_guesses = np.array([init_guesses])

    # 結果を格納するための空配列
    x = np.empty(init_guesses.shape)
    f = np.empty(init_guesses.shape[0])
    d = np.empty(init_guesses.shape[0], dtype=dict)

    # 各変数の上下界(min,max)を表すリストboundsを作成
    bounds = np.vstack((lower,upper)).T

    # すべての初期値について最適化を実行する
    for i, init_guess in enumerate(init_guesses):
        x[i], f[i], d[i] = spo.fmin_l_bfgs_b(fun, init_guess, bounds=bounds, approx_grad=True)

    # 最適解における評価関数の値の小さい順でソート
    x = x[np.argsort(f)]
    d = d[np.argsort(f)]
    f = np.sort(f)

#    if d[0]['warnflag'] != 0:
#        raise Exception("Not converged")

    # 最も良かった結果だけを返す
    return x[0], f[0], d[0]


# テスト
if __name__ == '__main__':

    from scipy.optimize import rosen

    def ackley(x):
        arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
        return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

    lower = np.array([-5,-5])
    upper = np.array([5,5])

    lower_ac = np.array([-32.768, -32.768])
    upper_ac = np.array([32.768, 32.768])

    res = multistart_minimizer(ackley, lower_ac, upper_ac, num_init=100)

    print(res[0])
    print(res[1])

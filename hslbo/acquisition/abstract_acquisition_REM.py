# -*- coding: utf-8 -*-
# abstract_acquisition_RE.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.optimize as spo

#from ..sampling.sobol import sobol_sequence

from ..model.gp import GP
from ..utils.multistart_minimizer import multistart_minimizer


DEFAULT_NUM_CANDIDATE = 300

DEFAULT_MCMC_ITERS = 10
DEFAULT_BURNIN     = 100

class AbstractAcquisitionREM(object):
    __metaclass__ = ABCMeta
    """ RamdomEmbeddingを用いた獲得関数のクラス """

    def __init__(self, num_dims, eff_dims, options={}):
        ### Effective dimensionality
        self.eff_dims = eff_dims

        ### Random matrix
        self.A = sps.norm.rvs(loc=0, scale=1, size=(self.num_dims, self.eff_dims))

        ### Set bound of efficient subspace
        self._set_subspace()

        # 入力の次元数
        self.num_dims = num_dims

        # 探索に用いる初期点の数
        self.num_candidate = int(options.get("num_candidate", DEFAULT_NUM_CANDIDATE))

        mcmc_iters = int(options.get("mcmc_iters", DEFAULT_MCMC_ITERS))
        burnin = int(options.get("burnin", DEFAULT_BURNIN))

        # ガウス過程回帰の初期化
        self.model = GP(self.num_dims, mcmc_iters=mcmc_iters, burnin=burnin)

        # オプションの代入
        self.set_options(options)

    def _set_subspace(self):
        """ Efficient subspaceの上下限を決定する """
        self.eff_lower = - np.srqt(self.eff_dims) * np.ones(self.eff_dims)
        self.eff_upper =   np.srqt(self.eff_dims) * np.ones(self.eff_dims)


    # メインの処理(基本はこれだけ呼び出す)
    def maximize(self):
        """ 獲得関数を最大化する """

        # 獲得関数が最大となるような点を見つける(L-BFGS法)
        # TODO: 他の大域的最適化アルゴリズムとの比較

        result_y = multistart_minimizer(self.acquisition_wrapper, self.eff_lower, self.eff_upper, init_guesses=None, num_init=self.num_candidate)

        # 元の高次元に埋め込み
        result_x = self._random_embedding(result_y[0])

        # 次の探索点を格納
        self.next_search = result_x

        # スケジューリングすべき変数があれば更新
        self.scheduling()

        return self.next_search


    ### ラッパーを用いて最適化問題を効率的次元で解く
    def acquisition_wrapper(self, y):
        """ 獲得関数のラッパー """
        x = self._random_embedding(y)
        # -1を掛けて最小化問題に変換
        return -1 * self.integrated_acquisition(x)

    def _random_embedding(self, y):
        """ Map from low to high dimensional space """
        # ランダムな線形写像により高次元へ埋め込む
        x = np.dot(self.A, y)
        x = self._bound(x)
        return x

    def _bound(self, x):
        """ box constraint [-1,1]^d から漏れた部分を埋め込む """
        one = np.ones(self.num_dims)
        x = np.amin(np.array([x, one]), axis=0)
        x = np.amax(np.array([x, -one]), axis=0)
        return x

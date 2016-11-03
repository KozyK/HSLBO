# -*- coding: utf-8 -*-
# abstract_acquisition.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.optimize as spo

#from ..sampling.sobol import sobol_sequence

from ..model.gp import GP
from ..utils.multistart_minimizer import multistart_minimizer


DEFAULT_NUM_CANDIDATE = 100

class AbstractAcquisition(object):
    __metaclass__ = ABCMeta

    def __init__(self, num_dims, lower, upper,  num_conditions=0, options={}):

        # 入力の次元数
        self.num_dims = num_dims
        # 条件変数の数
        self.num_conditions = num_conditions

        # 探索空間の下界
        self.lower = lower
        # 探索空間の上界
        self.upper = upper

        # 探索に用いる初期点の数
        self.num_candidate = int(options.get("num_candidate", DEFAULT_NUM_CANDIDATE))

        # ガウス過程回帰の初期化
        self.model = GP(self.num_dims)

        # オプションの代入
        self.set_options(options)

    @abstractmethod
    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        pass

#    def generate_init(self):
#        """ ソボル列を用いて初期値を発生，毎回変数のシャッフルを行う """
#        init_design = sobol_sequence(self.num_manipulate, self.num_candidate, loc=self.lower, scale=(self.upper-self.lower))
#        return init_design

    def fit(self, inputs, values):
        """ データを用いてガウス過程回帰モデルを更新する """
        self.model.fit(inputs, values)

    # メインの処理(基本はこれだけ呼び出す)
    def maximize(self, condition=None):
        """ 獲得関数を最大化する """

        # 条件変数があるときは格納する
        if self.num_conditions!=0:
            if not condition:
                raise Exception("condition variable is not specified ")
            self.condition = condition

        # 獲得関数が最大となるような点を見つける(L-BFGS法)
        # TODO: 他の大域的最適化アルゴリズムとの比較
        result = multistart_minimizer(self.acquisition_wrapper, self.lower, self.upper, init_guesses=None, num_init=self.num_candidate)

        # 次の探索点を格納
        self.next_search = result[0]

        # スケジューリングすべき変数があれば更新
        self.scheduling()

        return self.next_search

    def scheduling(self):
        """ パラメータのスケジューリング """
        pass

    def acquisition_wrapper(self, x):
        """ 獲得関数のラッパー """
        # 条件変数がある場合は操作変数に結合する
        if hasattr(self,"condition"):
            x = np.hstack((x,self.condition))
        # -1を掛けて最小化問題に変換
        return -1 * self.integrated_acquisition(x)

    def integrated_acquisition(self, x):
        """ ハイパーパラメータの事後分布で獲得関数を積分消去 """
        return self.model.function_over_hypers(self.acquisition, x)

    # TODO: ガウス過程が複数の出力を同時に計算できることを利用できていない → 効率化できる？

    # 獲得関数の計算
    @abstractmethod
    def acquisition(self, x):
        """ 獲得関数の計算 """
        pass

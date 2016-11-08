# -*- coding: utf-8 -*-
# rembo.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

import numpy as np
import scipy.stats as sps

#from pyDOE import lhs
#from .sampling.sobol import sobol_sequence

from .acquisition import ExpectedImprovement, MutualInformation, ConfidenceBound


ACQUISITION_FUNCTIONS = {
    'expected_improvement' : ExpectedImprovement,
    'confidence_bound' : ConfidenceBound,
    'mutual_information' : MutualInformation
}


class BO():
    """ Bayesian Optimization

    Parameters
    ----------
    func : callable
        function to be optimized
        最適化する関数
    num_dims : int
        The dimensionality of input space
        探索空間の次元数
    lower, upper : array-like (num_manipulate,)
        The lower and upper boundary of search space
        探索空間の上下界
    inputs : array-like, optional (num_samples, num_dims)
        過去の入力データ, default=None
    values : array-lke, optional (num_samples,)
        過去の出力データ, default=None
    criterion : String, optional
        獲得関数の選択
            expected_improvement : Expected Improvement(default),
            confidence_bound : (Lower) Confidence Bound,
            mutual_information : Mutual Information
    **options
        num_conditions : int
            The number of condition variables,
            条件変数（関数funcの入力のうち操作変数でない変数）の数．デフォルトは0.
        acq_options : dict
            Paramters for acquisition function
            獲得関数に渡すパラメータの辞書．デフォルトは空辞書
        mcmc_iters : int
            The number of MCMC samples
            MCMCで採用するサンプルの数．デフォルト10
        burnin : int
            The number of burnin samples
            バーンインで捨てるサンプルの数．デフォルト100


    TODO: 入力の正規化(操作変数と条件変数で変える？現時点では条件変数は既に正規化されていることを前提としている．)
    TODO: タスク定義の変更

    """

    # 内部では[-1,1]で格納し，出力時に[lower,upper]にする．

    def __init__(self, func, num_dims, lower, upper, inputs=None, values=None, **options):
        # 最小化する関数
        self.func = func

        # 探索空間の次元
        self._num_dims = num_dims
        # 探索空間の下界
        self._lower = lower
        # 探索空間の上界
        self._upper = upper
        # 探索空間の中心
        self._center = (lower + upper)/2
        # 探索空間のスケール
        self._scale = upper - lower

        # スケーリング済みの入出力データを格納
        self._inputs = np.empty((0,self.num_dims))
        self._values = np.empty((0,1))

        # 次の探索点を格納
        self._next_search = []

        self._num_conditions =options.get("num_conditions",0)

        # 出力変数の標準化
        #self.mean = options.get("mean", 0)
        #self.sigma = options.get("sigma", 1)

        acq_name = options.get("acq_name", "expected_improvement")

        # 獲得関数をセット
        self._set_acquisition(acq_name, options.get("acq_options", {}))

        # データを初期化
        self._initialize(inputs,values)


    def _scaling(self,inputs):
        """ 探索空間を[-1,1]の間にスケーリング """
        return 2 * (inputs - self._center) / self._scale


    def _descaling(self,inputs):
        """ 探索空間のスケールを復元する """
        return (1/2) * self._scale * inputs + self._center


    def _add_data(self, new_inputs, new_values):
        """ 入力・出力データを格納する """
        # データを格納する
        self._inputs = np.vstack((self._inputs, new_inputs))
        self._values = np.vstack((self._values, new_values))


    def _mod_to_2d(self, inputs):
        if inputs.ndim ==1:
            inputs = np.array([inputs])
        return inputs

    def _initialize(self, init_inputs=None, init_values=None):
        # もし過去のデータがあれば格納する
        if (not init_inputs is None) and (not init_values is None):
            print("Initial {0} designs are added.".format(values.size))
            # もし1次元なら2次元に変換
            init_inputs = self._mod_to_2d(init_inputs)
            # スケーリングを行う
            init_inputs = self._scaling(init_inputs)
            # すべてのデータを格納
            for x, y in zip(init_inputs, init_values):
                self._add_data(x, y)
        # なければランダムに初期実験計画を設定し評価
        else:
            print("No initial design.")
            # TODO:初期値の設定をどうするか.
            # 初期点を探索空間の中心に作成して評価
            next_search = np.zeros((1, self._num_dims))
            self._add_data(next_search, self._evaluate(next_search))

    # 指定した獲得関数をセットする
    def _set_acquisition(self, acq_name, acq_options):
        try:
            self.acquisition = ACQUISITION_FUNCTIONS[acq_name](self._num_dims, (-1)*np.ones(self._num_dims), np.ones(self._num_dims), num_conditions=self._num_conditions, options=acq_options)
        except:
            print("Specified acquisition does not exist.")

    def _evaluate(self, inputs, conditions=None):
        x = self._descaling(inputs)
        if conditions is not None:
            x = np.hstack((x, conditions))
        return self.func(np.squeeze(x))

    # 探索する
    def sequential_update(self, itertimes=1 ,condition=[]):
        if self._num_conditions == 0:
            for i in range(itertimes):
                self.update()
        else:
            for (i, condition) in zip(range(itertimes),conditions):
                self.update()

    #
    def update(self, condition=None):
        if self._num_conditions == 0:
            # 現在得られたデータでモデルを更新する
            self.acquisition.fit(self._inputs, self._values)
            # 獲得関数を最大化するself._next_searchを求める
            self._next_search = self.acquisition.maximize()
        else:
            self.acquisition.fit(self._inputs, self._values)
            # conditionの次元が適正か確認
            if condition.size != self.num_conditions:
                raise Exception("Condition variable does not have proper dimension.")
            # conditionのもとで獲得関数を最大化するself._next_searchを求める
            self._next_search = self.acquisition.maximize(condition=condition)

        # _next_searchにおける関数値を評価して格納
        self._add_data(self._next_search, self._evaluate(self._next_search))

    @property
    def history(self):
        """最良の観測値の履歴を返す"""
        opt_idx = []
        for i in range(self.values.size):
            opt_idx.append(np.argmin(self.values[:i+1]))
        x_hist = self.inputs[opt_idx]
        f_hist = self.values[opt_idx]
        return x_hist, f_hist

    @property
    def current_best(self):
        """ 最良の観測値を返す """
        idx_min = np.argmin(self._values, axis=0)
        return self.inputs[idx_min],  np.squeeze(self._values[idx_min])


    @property
    def model(self):
        """ 現時点で得られたモデルを返す """
        import copy
        model = self.acquisition.model.fit(self._inputs, self._values)
        return model.deepcopy()

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def num_samples(self):
        return self._inputs.shape[0]

    @property
    def inputs(self):
        return self._descaling(self._inputs[:,:self.num_dims])

    @property
    def conditions(self):
        return self._inputs[:,self.num_dims:]

    @property
    def values(self):
        return np.squeeze(self._values)

# -*- coding: utf-8 -*-
# slice_sampler.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.


import sys
import numpy as np
import numpy.random as npr

from .abstract_sampler import AbstractSampler
from ..utils import param as hyperparameter_utils


class SliceSampler(AbstractSampler):
    """ スライスサンプリングを用いてサンプルを生成

    Parameters
    ----------
    *params_to_sample : Param objects
        サンプリングを行うパラメータ
    **sampler_options

    Attributes
    ----------
    params : a list of Param objects
        リスト内の各要素の‘value’は‘self.sample()’を呼び出すたびに更新される
    """

    def logprob(self, x, model):
        """ 観測値xにおける対数確率を計算する

        これはパラメータの事前分布の確率とモデルの尤度を含む

        Returns
        -------
        lp : float
            対数確率
        """
        # self.paramsの中のパラメータにxを代入
        hyperparameter_utils.set_params_from_array(self.params, x)

        lp = 0.0

        # パラメータの事前分布の対数確率を合計
        for param in self.params:
            lp += param.prior_logprob()

            if np.isnan(lp):
                print("Param diagnostics:")
                param.print_diagnostics()
                print("Prior logprob: {0}".format(param.prior_logprob()))
                raise Exception("Prior returned {0} logprob".format(lp))

        # lpが無限ならば値を返す（これ以上足しても意味ないので）
        if np.isinf(lp):
            return lp

        # モデルの対数尤度を加える
        lp += model.log_likelihood()

        if np.isnan(lp):
            raise Exception("Likelihood returned {0} logprob".format(lp))

        return lp

    def sample(self, model):
        """ パラメータの新しいサンプルを生成 """
        params_array = hyperparameter_utils.params_to_array(self.params)
        for i in range(self.thinning + 1):
            # パラメータの新たな値をスライスサンプリングで取得
            params_array, current_ll = slice_sample(params_array, self.logprob, model, **self.sampler_options)
            hyperparameter_utils.set_params_from_array(self.params, params_array)
        self.current_ll = current_ll

def slice_sample(init_x, logprob, *logprob_args, **slice_sample_args):
    """ スライスサンプリングを用いて，確率密度関数から新たなサンプルを生成

    Paramters
    ---------
    init_x : array
        現在の位置
    logprob : callable, 'lprob = logprob(x, *logprob_args)'
        任意のxに対して対数確率密度を返す関数
    *logprob_args :
        additional arguments are passed to logprob

    Returns
    -------
    new_x : float
        新しい位置のサンプル
    new_llh : float
        新しい位置における対数確率

    Notes
    -----
    http://en.wikipedia.org/wiki/Slice_sampling

    """
    # step size
    sigma = slice_sample_args.get('sigma', 1.0)
    # step outする/しない
    step_out = slice_sample_args.get('step_out', True)
    # 最大step out回数
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    # 変数ごとに独立にサンプリング
    compwise = slice_sample_args.get('compwise', True)
    # 処理を出力
    verpose = slice_sample_args.get('verpose', False)

    # "direction"方向に1次元サンプリング
    def direction_slice(direction, init_x):

        def dir_logprob(z):
            try:
                return logprob(direction*z + init_x, *logprob_args)
            except:
                print('ERROR: Logprob failed at input {0}'.format(direction*z + init_x))

        # 現在のxの周りにUpper boundとLower boundを設定
        upper = sigma*npr.rand()
        lower = upper - sigma

        # [0, logprob(x)]で垂直方向に一様サンプリング
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        # step out回数をリセットする
        l_steps_out = 0
        u_steps_out = 0

        # step outを実行する
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma

        # 棄却サンプリング
        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper-lower)*npr.rand() + lower
            new_llh   = dir_logprob(new_z)

            # 対数尤度が計算可能でなければエラー
            if np.isnan(new_llh):
                print(new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x))
                raise Exception("Slice sampler got a NaN logprob")
            # 対数尤度が現在地点の尤度より大きければサンプリング
            if new_llh > llh_s:
                break
            # 対数尤度が現在地点の尤度より小さければshrink
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            # shrinkしても見つからなければエラー
            else:
                raise("Slice sampler shrank to zero")

        # 処理を全て書き出す
        if verpose:
            print("Steps Out: {0}, {1},  Step In: {2},  Final logprob: {3}".format(l_steps_out, u_steps_out, steps_in, new_llh))

        return new_z*direction + init_x, new_llh

    # 現在地点の対数尤度を評価
    initial_llh = logprob(init_x, *logprob_args)

    # サンプリングを行う前の対数尤度を出力
    if verpose:
        sys.stderr.write('Logprob before sampling: {0}'.format(initial_llh))
    # 対数尤度が無限小ならばエラー
    if np.isneginf(initial_llh):
        sys.stderr.write('Values passed into slice sampler:'.format(init_x))
        raise Exception("Initial value passed into slice sampler has logprob = -inf")

    # 初期値が1次元配列ならば2次元配列に変換
    if not init_x.shape:
        init_x = np.array([init_x])

    # 変数の次元
    dims = init_x.shape[0]

    # 変数ごとに独立にサンプリング
    if compwise:
        # サンプリングを行う順番を決定
        ordering = np.arange(dims)
        npr.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            # [0, ..., 0, 1, 0, ..., 0] 方向に探索
            direction = np.zeros((dims))
            direction[d] = 1.0
            cur_x, cur_llh = direction_slice(direction, cur_x)

    # ランダムな方向にサンプリング
    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        cur_x, cur_llh = direction_slice(direction, init_x)

    return cur_x, cur_llh

# -*- coding: utf-8 -*-
# gp.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

import numpy as np
from functools import reduce

from abc import ABCMeta, abstractmethod

class AbstractModel(object):
    """ モデルの抽象クラス """
    __metaclass__ = ABCMeta

    # 辞書にパラメータの値を格納
    @abstractmethod
    def to_dict(self):
        pass

    # 辞書からパラメータの値を取り出す
    @abstractmethod
    def from_dict(self):
        pass

    # モデルをデータにフィッティングしてハイパーパラメタを取得
    @abstractmethod
    def fit(self, inputs, values, meta=None, hypers=None):
        pass

    # モデルの対数尤度を返す
    @abstractmethod
    def log_likelihood(self):
        pass

    # 新規入力に対して出力の予測平均と予測分散を返す
    @abstractmethod
    def predict(self, pred, full_cov=False, compute_grad=False):
        pass


    def function_over_hypers(self, fun, *fun_args, **fun_kwargs):
        """ 格納されているハイパーパラメータに対応するモデルを用いて
            計算される関数fun(獲得関数etc)の平均をとる"""
        return function_over_hypers([self], fun, *fun_args, **fun_kwargs)

def function_over_hypers(models, fun, *fun_args, **fun_kwargs):
    """ 格納されているハイパーパラメータに対応するモデル(複数)を用いて
        計算される関数fun(獲得関数etc)の平均をとる

    If models have different numbers of samples, use the first n samples of each model, where n is the min number of samples over the models
    """
    # modelsの中でもっとも少ない状態の数に合わせる
    min_num_states = reduce(min, map(lambda x: x.num_states, models), np.inf)

    # ハイパーパラメータの各状態をself.paramsに順に格納
    for i in range(min_num_states):

        # modelsに含まれる，全てのmodelについて
        # ハイパーパラメータをi番目の状態にセット
        for model in models:
            model.set_state(i)

        # i番目の状態において関数の値を計算する
        result = fun(*fun_args, **fun_kwargs)

        # 最初の状態で関数の出力が単数か複数かを調べ
        # 関数の値を格納するための空のnumpy配列(のリスト)を用意
        if i == 0:
            if type(result) is tuple:
                isTuple = True
                average = [np.zeros(r.shape) for r in result]
            else:
                isTuple = False
                average = np.zeros(result.shape)

        # 結果を格納する
        if isTuple:
            # 出力の数が不正なら警告
            assert(len(result) == len(average))
            for j in range(len(average)):
                # j番目の出力のサイズが不正ならば警告
                assert(result[j].shape == average[j].shape)
                # j番目の出力を加算
                average[j] += result[j]
        else:
            # 出力のサイズが不正なら警告
            assert(result.shape == average.shape)
            # 出力を加算
            average += result

    # 状態の数で割って平均をとる
    if isTuple:
        for j in range(len(average)):
            average[j] /= min_num_states
    else:
        average /= min_num_states

    return average

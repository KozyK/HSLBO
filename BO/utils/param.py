# -*- coding: utf-8 -*-
# param.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

import copy

import numpy as np

from . import priors

def set_params_from_array(params_iterable, params_array):
    """ params_iterableの中のパラメータをparams_arrayに格納されている新たな値で更新．
        params_arrayはパラメータの値が格納されたnumpy配列
    """
    index = 0
    for param in params_iterable:
        if param.size() == 1 and not param.isArray:
            param.value = params_array[index]
        else:
            param.value = params_array[index:index+param.size()]
        index += param.size()

def params_to_array(params_iterable):
    """params_iterableに格納されているparamの値をサンプリングのために1次元配列に格納"""
    return np.hstack([param.value for param in params_iterable])

#def params_to_dict(params_iterable):
#    """params_iterableに格納されているparamで辞書型のparams_dictの値を更新する"""
#    params_dict = {}
#    for param in params_iterable:
#        params_dict[param.name] = param.value
#    return params_dict

# def params_to_compressed_dict():
#    return

# パラメータをクラスとして定義
class Param(object):
    """ パラメータを表すクラス
    """

    def __init__(self, initial_value, prior=priors.NoPrior(), name="Unnamed"):
        # パラメータの初期値
        self.initial_value = copy.copy(initial_value)
        #️ パラメータの現在の値
        self.value         = initial_value
        # パラメータの名前
        self.name          = name
        # パラメータの事前分布
        self.prior         = prior
        # initial_valueがnumpy配列ならば，そのことを記憶
        self.isArray       = hasattr(initial_value, "shape") and initial_value.shape != ()

    # パラメータの値を変更
    def set_value(self, new_value):
        self.value = new_value

    # パラメータの値をリセット
    def reset_value(self):
        self.value = self.initial_value

    # パラメータの値(i番目)を取得
    def get_value(self, i):
        if i < 0 or i >= self.size():
            raise Exception("param {0}: {1} out of bounds, size={2}".format(self.name, i, self.size()))
        if self.isArray:
            return self.value[i]
        else:
            return self.value

    # パラメータが配列ならば，サイズを取得
    def size(self):
        try:
            return self.value.size
        except:
            return 1

    # 現在の値における事前分布の対数確率密度
    def prior_logprob(self):
        return self.prior.logprob(self.value)

    # パラメータを事前分布からサンプリングする
    def sample_from_prior(self):
        if hasattr(self.prior, 'sample'):
            self.value = self.prior.sample(self.size())
        else:
            raise Exception("Param {0} has prior {1}, which does not allow sampling".format(self.name, self.prior.__class__.__name__))
        """
        try:
            self.value = float(value)
        except:
            pass
        """

    # パラメータの名前と値を返す．(配列ならば最大・最小値とサイズを返す)
    def print_diagnostics(self):
        if self.size() == 1:
            print("{0}: {1}".format(self.name, self.value))
        else:
            print("{0}: min={1}, max={2} (size={3})".format(self.name, self.value.min(), self.value.max(), self.size()))

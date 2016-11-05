# -*- coding: utf-8 -*-
# abstract_sampler.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

from abc import ABCMeta, abstractmethod

from ..utils import param as hyperparameter_utils


class AbstractSampler(object):
    """ サンプラーの抽象クラス """
    __metaclass__ = ABCMeta

    def __init__(self, *params_to_sample, **sampler_options):
        # サンプリングをするパラメータを代入
        self.params          = params_to_sample
        # サンプラーのオプションを代入
        self.sampler_options = sampler_options
        # 現在の対数尤度
        self.current_ll      = None
        # sampler_optionsに'thinning'がなければTrue
        self.thinning_overridable = not 'thinning' in sampler_options
        # thinning(間伐)を行う間隔を決める．間伐はMCMCの標本の独立性を担保するために行うが，あまり良い手法ではない？
        self.thinning        = sampler_options.get('thinning', 0)



    @abstractmethod
    def logprob(self, x, model):
        pass

    @abstractmethod
    def sample(self, model):
        pass

    # サンプリングするパラメータの中身を調べる
    def print_diagnostics(self):
        params_array = hyperparameter_utils.params_to_array(self.params)
        for params in self.params:
            param.print_diagnostics()

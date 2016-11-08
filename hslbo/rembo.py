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

from . import BO
from .acquisition import ExpectedImprovementREM, MutualInformationREM, ConfidenceBoundREM

ACQUISITION_FUNCTIONS = {
    'expected_improvement' : ExpectedImprovementREM,
    'confidence_bound' : ConfidenceBoundREM,
    'mutual_information' : MutualInformationREM
}


class REMBO(BO):
    """ Bayesian Optimization with Random Embedding

    Additional Parameters
    ---------------------
    eff_dims : int
        The dimensionality of efficient subspace
        効率的に探索を行える空間の次元数
    """


    def __init__(self, func, num_dims, lower, upper, eff_dims, inputs=None, values=None, **options):

        ### 探索空間の次元
        self._eff_dims = eff_dims

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

    # 指定した獲得関数をセットする
    def _set_acquisition(self, acq_name, acq_options):
        try:
            self.acquisition = ACQUISITION_FUNCTIONS[acq_name](self._num_dims, self.eff_dims, options=acq_options)
        except:
            print("Specified acquisition does not exist.")

# -*- coding: utf-8 -*-
# confidence_bound.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

from .abstract_acquisition import AbstractAcquisition

import numpy as np
import scipy.stats as sps

class ConfidenceBound(AbstractAcquisition):
    """ GP Lower Confidence Bound

    Additional Parameters
    ---------------------
    beta : float


    Reference
    ---------
    Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).

    """

    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        self.kappa = options.get("kappa", 0.1)

    def acquisition(self, x):
        """ 獲得関数の計算 """
        mu, var = self.model.predict(x)
        sigma = np.sqrt(var)

        lcb =  - (mu - self.kappa * sigma)

        return cb

    def scheduling(self):
        """ パラメータkappaの更新 """
        pass

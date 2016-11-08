# -*- coding: utf-8 -*-
# expected_improvement.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

from .abstract_acquisition import AbstractAcquisition

import numpy as np
import scipy.stats as sps

class ExpectedImprovement(AbstractAcquisition):
    """ Expected Improvement

    Reference
    ---------
    Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).

    """

    def fit(self, inputs, values):
        """ データを用いてガウス過程回帰モデルを更新する """
        # ターゲットを最適値に設定
        self.model.fit(inputs, values)
        estimated_values, _ = self.model.predict(inputs)
        self.target = np.amin(estimated_values)

    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        pass

    def acquisition(self, x):
        """ 獲得関数の計算 """
        mu, var = self.model.predict(x)
        sigma = np.sqrt(var)
        z = (self.target - mu)/sigma
        fi = sps.norm()
        return sigma * z * fi.cdf(z) + fi.pdf(z)

class ExpectedImprovementREM(AbstractAcquisitionREM):
    """ Expected Improvement with Random Embedding

    Reference
    ---------
    Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).

    """

    def fit(self, inputs, values):
        """ データを用いてガウス過程回帰モデルを更新する """
        # ターゲットを最適値に設定
        self.model.fit(inputs, values)
        estimated_values, _ = self.model.predict(inputs)
        self.target = np.amin(estimated_values)

    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        pass

    def acquisition(self, x):
        """ 獲得関数の計算 """
        mu, var = self.model.predict(x)
        sigma = np.sqrt(var)
        z = (self.target - mu)/sigma
        fi = sps.norm()
        return sigma * z * fi.cdf(z) + fi.pdf(z)

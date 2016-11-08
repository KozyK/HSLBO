# -*- coding: utf-8 -*-
# mutual_information.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

from .abstract_acquisition import AbstractAcquisition

import numpy as np
import scipy.stats as sps

class MutualInformation(AbstractAcquisition):
    """ Mutual Information

    Additional Parameters
    ---------------------
    sigma : float (0, 1)
        confidence parameter
    alpha : float
        tradeoff parameter
        sigmaによって間接的に決められるが直接指定も可能

    Reference
    ---------
    Perchet, V. (2014). Gaussian process optimization with mutual information.

    """

    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        self.gamma = 0

        self.sigma = options.get("sigma", 1.0e-1)
        if (self.sigma < 0.0) or (1.0 < self.sigma):
            raise Exception("sigma is out of boundary")

        self.alpha = options.get("alpha", np.log(2/self.sigma))

    def acquisition(self, x):
        """ 獲得関数の計算 """
        mu, var = self.model.predict(x)

        fi = np.sqrt(self.alpha) * (np.sqrt(var + self.gamma) - np.sqrt(self.gamma))

        mi = - mu + fi

        return mi

    def scheduling(self):
        """ パラメータgammaの更新 """
        _, var = self.model.predict(self.next_search)
        self.gamma = self.gamma + var


class MutualInformationREM(AbstractAcquisitionREM):
    """ Mutual Information with Random Embedding

    Additional Parameters
    ---------------------
    sigma : float (0, 1)
        confidence parameter
    alpha : float
        tradeoff parameter
        sigmaによって間接的に決められるが直接指定も可能

    Reference
    ---------
    Perchet, V. (2014). Gaussian process optimization with mutual information.

    """

    def set_options(self, options):
        """ 獲得関数のパラメータの代入 """
        self.gamma = 0

        self.sigma = options.get("sigma", 1.0e-1)
        if (self.sigma < 0.0) or (1.0 < self.sigma):
            raise Exception("sigma is out of boundary")

        self.alpha = options.get("alpha", np.log(2/self.sigma))

    def acquisition(self, x):
        """ 獲得関数の計算 """
        mu, var = self.model.predict(x)

        fi = np.sqrt(self.alpha) * (np.sqrt(var + self.gamma) - np.sqrt(self.gamma))

        mi = - mu + fi

        return mi

    def scheduling(self):
        """ パラメータgammaの更新 """
        _, var = self.model.predict(self.next_search)
        self.gamma = self.gamma + var

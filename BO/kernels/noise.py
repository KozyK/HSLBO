# -*- coding: utf-8 -*-
# noise.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.


import numpy as np

from .abstract_kernel import AbstractKernel
from ..utils          import priors
from ..utils.param    import Param as Hyperparameter


class Noise(AbstractKernel):
    def __init__(self, num_dims, noise=None, name='Noise'):
        self.name     = name
        self.num_dims = num_dims

        default_noise = Hyperparameter(
            initial_value = 1e-6,
            prior         = priors.NonNegative(priors.Horseshoe(0.1)),
            name          = 'noise'
        )

        self.noise = noise if noise is not None else default_noise

    @property
    def hypers(self):
        return self.noise

    def cov(self, inputs):
        return np.diag(self.noise.value*np.ones(inputs.shape[0]))

    def diag_cov(self, inputs):
        return self.noise.value*np.ones(inputs.shape[0])

    def cross_cov(self, inputs_1, inputs_2):
        return np.zeros((inputs_1.shape[0],inputs_2.shape[0]))

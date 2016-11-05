# -*- coding: utf-8 -*-
# matern.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

import numpy as np
from scipy.spatial.distance import cdist

from ..utils                import priors
from ..utils.param          import Param as Hyperparameter

from .abstract_kernel import AbstractKernel

# 重み付き距離の計算
# lsはlength scale (1*D)
def dist2(ls, x1, x2=None):
    # 2点間の重み付き距離の2乗を計算
    # Assumes N*D and M*D matrices
    if x2 is None:
        x2 = x1
    xx1 = x1/ls
    xx2 = x2/ls
    r2 = cdist(xx1, xx2, 'sqeuclidean')
    return r2

SQRT_5 = np.sqrt(5)
SQRT_3 = np.sqrt(3)

class Matern52(AbstractKernel):
    """calculate ARD Matern 5/2 kernel with gaussian noise.
    """
    def __init__(self, num_dims, length_scale=None, name='Matern52'):
        self.name = name
        self.num_dims = num_dims

        default_ls = Hyperparameter(
            initial_value = np.ones(self.num_dims),
            prior         = priors.Tophat(0,10),
            name          = 'ls'
        )

        self.ls = length_scale if length_scale is not None else default_ls

        assert self.ls.value.shape[0] == self.num_dims

    @property
    def hypers(self):
        return self.ls

    def cov(self, inputs):
        return self.cross_cov(inputs, inputs)

    def diag_cov(self, inputs):
        return np.ones(inputs.shape[0])

    def cross_cov(self, inputs_1, inputs_2):
        r2 = np.abs(dist2(self.ls.value, inputs_1, inputs_2))
        r = np.sqrt(r2)
        cov = (1.0 + SQRT_5*r + (5.0/3.0)* r2) * np.exp(-SQRT_5*r)
        return cov

    def cross_cov_grad_data(self, inputs_1, inputs_2):
        # NOTE: This is the gradient wrt the inputs of inputs_2
        # The gradient wrt the inputs of inputs_1 is -1 times this
        r2      = np.abs(kernel_utils.dist2(self.ls.value, inputs_1, inputs_2))
        r       = np.sqrt(r2)
        grad_r2 = (5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)

        return grad_r2[:,:,np.newaxis] * kernel_utils.grad_dist2(self.ls.value, inputs_1, inputs_2)

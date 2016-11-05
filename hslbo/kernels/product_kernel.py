# -*- coding: utf-8 -*-
# product_kernel.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.



import numpy as np

from .abstract_kernel import AbstractKernel


class ProductKernel(AbstractKernel):
    # TODO: If all kernel values are positive then we can do things in log-space

    def __init__(self, *kernels):
        self.kernels = kernels

    def cov(self, inputs):
        return reduce(lambda K1, K2: K1*K2, [kernel.cov(inputs) for kernel in self.kernels])

    def diag_cov(self, inputs):
        return reduce(lambda K1, K2: K1*K2, [kernel.diag_cov(inputs) for kernel in self.kernels])

    def cross_cov(self, inputs_1, inputs_2):
        return reduce(lambda K1, K2: K1*K2, [kernel.cross_cov(inputs_1,inputs_2) for kernel in self.kernels])

    # This is the gradient wrt **inputs_2**
    def cross_cov_grad_data(self, inputs_1, inputs_2):
        vals  = np.array([kernel.cross_cov(inputs_1,inputs_2) for kernel in self.kernels])
        vprod = reduce(lambda x, y: x*y, vals)
        grads = np.array([kernel.cross_cov_grad_data(inputs_1,inputs_2) for kernel in self.kernels])
        V     = vals == 0

        return (((vprod[:,:,np.newaxis]*grads) / (vals + V)[:,:,:,np.newaxis]) + (V[:,:,:,np.newaxis]*grads)).sum(0)

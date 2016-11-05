# -*- coding: utf-8 -*-
# sum_kernel.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

from .abstract_kernel import AbstractKernel
from functools import reduce

class SumKernel(AbstractKernel):
    def __init__(self, *kernels):
        self.kernels = kernels

    def cov(self, inputs):
        return reduce(lambda K1, K2: K1+K2, [kernel.cov(inputs) for kernel in self.kernels])

    def diag_cov(self, inputs):
        return reduce(lambda K1, K2: K1+K2, [kernel.diag_cov(inputs) for kernel in self.kernels])

    def cross_cov(self, inputs_1, inputs_2):
        return reduce(lambda K1, K2: K1+K2, [kernel.cross_cov(inputs_1,inputs_2) for kernel in self.kernels])

    def cross_cov_grad_data(self, inputs_1, inputs_2):
        return reduce(lambda dK1, dK2: dK1+dK2, [kernel.cross_cov_grad_data(inputs_1,inputs_2) for kernel in self.kernels])

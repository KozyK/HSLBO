# -*- coding: utf-8 -*-
# abstract_kernel.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

import numpy as np

from abc import ABCMeta, abstractmethod

class AbstractKernel(object):
    __metaclass__ = ABCMeta

    @property
    def hypers(self):
        return None

    @abstractmethod
    def cov(self, inputs):
        pass

    @abstractmethod
    def diag_cov(self, inputs):
        pass

    @abstractmethod
    def cross_cov(self, inputs_1, inputs_2):
        pass

    @abstractmethod
    def cross_cov_grad_data(self, inputs_1, inputs_2):
        pass

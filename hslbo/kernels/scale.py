# -*- coding: utf-8 -*-
# scale.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.


from .abstract_kernel import AbstractKernel
from ..utils          import priors
from ..utils.param    import Param as Hyperparameter


class Scale(AbstractKernel):
    def __init__(self, kernel, amp2=None, name='ScaleKernel'):
        self.name   = name
        self.kernel = kernel

        default_amp2 = Hyperparameter(
            initial_value = 1.0,
            prior         = priors.LognormalOnSquare(1.0),
            name          = 'amp2'
        )

        self.amp2 = amp2 if amp2 is not None else default_amp2

    @property
    def hypers(self):
        return self.amp2

    def cov(self, inputs):
        return self.amp2.value*self.kernel.cov(inputs)

    def diag_cov(self, inputs):
        return self.amp2.value*self.kernel.diag_cov(inputs)

    def cross_cov(self, inputs_1, inputs_2):
        return self.amp2.value*self.kernel.cross_cov(inputs_1,inputs_2)

    # This is the gradient wrt **inputs_2**
    def cross_cov_grad_data(self, inputs_1, inputs_2):
        return self.amp2.value*self.kernel.cross_cov_grad_data(inputs_1,inputs_2)

# -*- coding: utf-8 -*-
# priors.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as npr
import scipy.stats as sps

class AbstractPrior(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def logprob(self, x):
        pass

class Tophat(AbstractPrior):
    def __init__(self, xmin, xmax):
            self.xmin = xmin
            self.xmax = xmax
            if not (xmax > xmin):
                raise Exception("xmax must be greater than xmin")

    def logprob(self, x):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
            return -np.inf
        else:
            return 0. # More correct is -np.log(self.xmax-self.xmin),but constants don't matter

    def sample(self, n_samples):
        return self.xmin + npr.rand(n_samples) * (self.xmax-self.xmin)

class Horseshoe(AbstractPrior):
    def __init__(self, scale):
        self.scale = scale

    # THIS IS INEXACT
    def logprob(self, x):
        if np.any(x == 0.0):
            return np.inf  # POSITIVE infinity (this is the "spike")
        # We don't actually have an analytical form for this
        # But we have a bound between 2 and 4, so I just use 3.....
        # (or am I wrong and for the univariate case we have it analytically?)
        return np.sum(np.log(np.log(1 + 3.0 * (self.scale/x)**2) ) )

    def sample(self, n_samples):
        # Sample from standard half-cauchy distribution
        lamda = np.abs(npr.standard_cauchy(size=n_samples))

        # I think scale is the thing called Tau^2 in the paper.
        return npr.randn() * lamda * self.scale

class Lognormal(AbstractPrior):
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean

    def logprob(self, x):
        return np.sum(sps.lognorm.logpdf(x, self.scale, loc=self.mean))

    def sample(self, n_samples):
        return npr.lognormal(mean=self.mean, sigma=self.scale, size=n_samples)

# 対数正規分布の2乗の従う分布
class LognormalOnSquare(Lognormal):
    def logprob(self, y):
        if np.any(y < 0): # Need ths here or else sqrt may occur with y < 0
            return -np.inf
        x = np.sqrt(y)
        dy_dx = 2*x
        return Lognormal.logprob(self, x) - np.log(dy_dx)

    def sample(self, n_samples):
        return Lognormal.sample(self, n_samples)**2

class Gaussian(AbstractPrior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def logprob(self, x):
        return np.sum(sps.norm.logpdf(x, loc=self.mu, scale=self.sigma))
            # equivalent to below
            # import math
            # return -(1/2)*(2*math.pi*(self.sigma**2)) - np.sum((np.power((x - self.mu), 2)) / (2 * self.sigma**2)
    def sample(self, n_samples):
        return sps.norm.rvs(loc=self.mu, scale=self.sigma, size=n_samples)

class MultivariateGaussian(AbstractPrior):
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        if mu.size != cov.shape[0] or cov.shape[0] != cov.shape[1]:
            raise Exception("Mu must be a vector and cov a matrix, of matching sizes") # エラー分岐これだけでいい？共分散行列の正定値性とか対称性とか
        def logprob(self, x):
            return sps.multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov)

        def sample(self, n_samples):
            return sps.multivariate_normal.rvs(mean=self.mu, cov=self.cov, size=n_samples).T.squeeze()

class NoPrior(AbstractPrior):
    def __init__(self):
        pass

    def logprob(self,x):
        return 0.0

# This class takes in another prior in its constructor
# And gives you the nonnegative version (actually the positive version, to be numerically safe)
class NonNegative(AbstractPrior):
    def __init__(self, prior):
        self.prior = prior

        if hasattr(prior, 'sample'):
            self.sample = lambda n_samples: np.abs(self.prior.sample(n_samples))

    def logprob(self, x):
        if np.any(x <= 0):
            return -np.inf
        else:
            return self.prior.logprob(x)# + np.log(2.0)
        # Above: the log(2) makes it correct, but we don't ever care about it I think


# This class allows you to compose a list of priors
# (meaning, take the product of their PDFs)
# The resulting distribution is "improper" --i.e. notnormalized
class ProductOfPriors(AbstractPrior):
    def __init__(self, priors):
        self.priors = priors

    def logprob(self, x):
        lp = 0.0
        for prior in self.priors:
            lp += prior.logprob(x)
        return lp

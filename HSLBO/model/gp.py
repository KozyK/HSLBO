# -*- coding: utf-8 -*-
# gp.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human Systems Laboratory, Department of Systems Science,
# Graduate school of Informatics, Kyoto University.

import sys
import logging
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps

from .abstract_model    import AbstractModel
from ..utils.param      import Param as Hyperparameter
from ..kernels          import Matern52, Noise, Scale, SumKernel
from ..sampling.slice_sampler import SliceSampler
from ..utils            import priors


DEFAULT_MCMC_ITERS = 10
DEFAULT_BURNIN     = 1000

class GP(AbstractModel):
    """ ガウス過程回帰

    Parameters
    ----------

    num_dims : int
        入力変数の次元
    likelihood ; str, optional
        尤度関数の選択
        'noiseless'もしくは'gaussian'
        デフォルトは'gaussian'
    verpose : bool, optional

    mcmc_diagnostics : bool, optional
        MCMCの診断
    mcmc_iters : bool, optional
        MCMCの繰り返し回数
    burnin : int, optional
        焼きなまし回数
    thinning : int, optional
        間伐回数
    """
    def __init__(self, num_dims, **options):
        self.num_dims = num_dims
        self._set_likelihood(options)

        self.verpose          = bool(options.get("verpose", False))
        self.mcmc_diagnostics = bool(options.get("mcmc_diagnostics", False))
        self.mcmc_iters       = int(options.get("mcmc_iters", DEFAULT_MCMC_ITERS))
        self.burnin           = int(options.get("burnin", DEFAULT_BURNIN))
        self.thinning = int(options.get("thinning", 0))

        # 入力データを格納
        self._inputs = None
        # 出力データを格納
        self._values = None
        # ハイパーパラメータの値を格納
        self.params = None
        # コレスキー分解を事前に行う/行わない
        self._caching = bool(options.get("caching", True))
        # コレスキー分解のalpha,cholを格納するリスト
        self._cache_list = []
        # ハイパーパラメータのリスト
        self._hypers_list = []
        # シードを格納
        self._random_state = npr.get_state()
        # サンプラーを格納
        self._samplers = []
        # カーネル
        self._kernel = None
        # 観測ノイズ込みのカーネル
        self._kernel_with_noise = None
        # サンプリングを行って得た状態の数
        self.num_states   = 0
        # マルコフ連鎖の長さ
        self.chain_length = 0

        self._build()


    def _set_likelihood(self, options):
        self.likelihood = options.get('likelihood', 'gaussian').lower()

        if self.likelihood == "noiseless":
            self.noiseless = True
        else:
            self.noiseless = False

    # 辞書型のhypers_dictから
    #️ 同じキーをもつ値を辞書型のself.paramsに代入
    def _set_params_from_dict(self, hypers_dict):
        for name, hyper in self.params.items():
            self.params[name].value = hypers_dict[name]

    # パラメータをリセット
    def _reset_params(self):
        for param in self.params.values():
            param.value = param.initial_value

    # self.cachingがTrueの時にコレスキー分解を計算
    def _pull_from_cache_or_compute(self):
        # self._cache_list の長さが状態の数と一致するときには計算済み
        if self.caching and len(self._cache_list) == self.num_states:
            chol = self._cache_list[self.state]['chol']
            alpha = self._cache_list[self.state]['alpha']
        # そうでない場合はその場で計算する
        else:
            chol = spla.cholesky(self.kernel.cov(self.inputs), lower=True)
            alpha = spla.cho_solve((chol, True), self.values - self.mean.value)
        return chol,alpha


    # コレスキー分解を行って_cache_listに保存
    def _prepare_cache(self):
        for i in range(self.num_states):
            self.set_state(i)
            chol  = spla.cholesky(self.kernel.cov(self.inputs), lower=True)
            alpha = spla.cho_solve((chol, True), self.values - self.mean.value)
            cache_dict = {
                'chol'  : chol,
                'alpha' : alpha
            }
            self._cache_list.append(cache_dict)

    #
    def _reset(self):
        self._cache_list = []
        self.hypers_list = []
        self._reset_params()
        self.chain_length = 0

    def _build(self):
        # ARD Matern 5/2 Kernel
        input_kernel           = Matern52(self.num_dims)
        # Noise Kernel: 計算安定性のため微小なノイズを付加
        stability_noise_kernel = Noise(self.num_dims)
        # Amplitude
        scaled_input_kernel    = Scale(input_kernel)
        # Matern 5/2 Kernel + Noise Kernel
        self._kernel             = SumKernel(scaled_input_kernel, stability_noise_kernel)
        # Noise Kernel: 観測ノイズ
        noise_kernel           = Noise(self.num_dims)


        # 観測ノイズを過程した場合，さらに結合
        if not self.noiseless:
            self._kernel_with_noise = SumKernel(self._kernel, noise_kernel)

        # 平均値を表すハイパーパラメータ
        self.mean = Hyperparameter(
            initial_value = 0.0,
            prior        = priors.Gaussian(0.0, 1.0),
            name          = 'mean'
        )

        # サンプリングのためにハイパーパラメータを取得
        ls                 = input_kernel.hypers
        amp2               = scaled_input_kernel.hypers

        # ハイパーパラメータの辞書
        self.params = {
            'mean'      : self.mean,
            'amp2'      : amp2,
            'ls'        : ls
        }

        # 観測ノイズがあれば辞書に追加してサンプラー構築
        if self.noiseless:
            self._samplers.append(SliceSampler(self.mean, amp2, compwise=False, thinning=self.thinning))
        else:
            noise = noise_kernel.hypers
            self.params.update({'noise' : noise})
            self._samplers.append(SliceSampler(self.mean, amp2, noise, compwise=False, thinning=self.thinning))
        # 変数のスケールを表すパラメータはcompwiseにサンプリング
        self._samplers.append(SliceSampler(ls, compwise=True, thinning=self.thinning))

    # マルコフ連鎖モンテカルロ：焼きなまし
    def _burn_samples(self, num_samples):
        for i in range(num_samples):
            for sampler in self._samplers:
                sampler.sample(self)
            # マルコフ連鎖の長さ ＋1
            self.chain_length += 1

    # マルコフ連鎖モンテカルロ：サンプリングを行い，
    # self.paramsの値を更新．
    # 結果を集めてhypers_listとして返す
    def _collect_samples(self, num_samples):
        hypers_list = []
        for i in range(num_samples):
            for sampler in self._samplers:
                sampler.sample(self)
            hypers_list.append(self.to_dict()['hypers'])
            # マルコフ連鎖の長さ ＋1
            self.chain_length += 1
        return hypers_list

    @property
    def inputs(self):
        return self._inputs

    @property
    def observed_inputs(self):
        return self._inputs

    @property
    def values(self):
        return self._values

    @property
    def observed_values(self):
        return self._values

    @property
    def kernel(self):
        if self.noiseless:
            return self._kernel
        else:
            return self._kernel_with_noise if self._kernel_with_noise is not None else self._kernel

    @property
    def noiseless_kernel(self):
        return self._kernel

    @property
    def has_data(self):
        return self.inputs is not None

    @property
    def caching(self):
        if not self._caching or self.num_states <= 0:
            return False
        return True

    # 指定したstateのパラメータ候補を取り出してself.paramsに格納
    def set_state(self, state):
        self.state = state
        self._set_params_from_dict(self._hypers_list[state])

    # 辞書にハイパーパラメータの値とマルコフ連鎖の長さを格納
    def to_dict(self):
        gp_dict = {'hypers' : {}}
        for name, hyper in self.params.items():
            gp_dict['hypers'][name] = hyper.value
        gp_dict['chain length'] = self.chain_length
        return gp_dict

    # 辞書からハイパーパラメータの値をself.paramsに格納し，
    # マルコフ連鎖の値をself.chain_lengthに格納
    def from_dict(self, gp_dict):
        self._set_params_from_dict(gp_dict['hypers'])
        self.chain_length = gp.dict['chain length']


    def fit(self, inputs, values, hypers=None, reburn=False, fit_hypers=True):
        """ガウス過程モデルを入出力データにフィッティングしたのちに
        ハイパーパラメータの値を返す．

        Parameters
        ----------
        inputs : 2d array
            入力データ
        values : 1d array
            出力データ
        hypers : dict
            ハイパーパラメータの初期値
        reburn : bool
            再度焼きなましするか否か(デフォルトはFalse)
        fit_hypers : bool
            ハイパーパラメータのフィッティングを行う/行わない(デフォルトはTrue)
        """
        # データを格納
        self._inputs = inputs
        self._values = values

        # モデルをリセット
        self._reset()

        # ハイパーパラメータが指定されたときにself.paramsに格納
        if hypers:
            self.from_dict(hypers)

        # ハイパーパラメータのフィッティングを行う
        if fit_hypers:
            # reburn=Trueもしくはマルコフ連鎖の長さがburn in期間より短いならばnum_samplesの数だけburn inを行う
            num_samples = self.burnin if reburn or self.chain_length < self.burnin else 0
            self._burn_samples(num_samples)
            # self.mcmc_itersの回数だけサンプリングを実施して
            # self._hypers_listに格納
            self._hypers_list = self._collect_samples(self.mcmc_iters)
            # 得られた状態の数を更新
            self.num_states = self.mcmc_iters

        # ハイパーパラメータのフィッティングを行わず，指定した単一の値を利用する
        elif not self._hypers_list:
            self._hypers_list = [self.to_dict()['hypers']]
            self.num_states = 1

        # 全てのパラメータ候補に対してコレスキー分解を行いself._cache_listに保存
        if self.caching:
            self._prepare_cache()

        # 現時点でのマルコフ連鎖の終端をself.paramsに格納
        self.set_state(len(self._hypers_list)-1)

        # 終端のパラメータ候補とマルコフ連鎖の長さを返す．
        # gp_dict={"hypers":~,"chain length":~}
        # return self.to_dict() # 返り値は返さなくていいかも


    def log_likelihood(self):
        """ ガウス過程の周辺尤度 """
        cov = self.kernel.cov(self.observed_inputs)
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), self.observed_values - self.mean.value)

        return -np.sum(np.log(np.diag(chol)))-0.5*np.dot((self.observed_values - self.mean.value).T, solve)


    def predict(self, pred, full_cov=False, compute_grad=False):
        """ 新規入力データに対して，平均値と分散の予測値を返す

        Parameters
        ----------
        pred : 2d array
            予測を行う入力データの配列
        full_cov : bool
            共分散行列を対角成分以外も含めてすべて計算する(デフォルト:False)
        compute_grad : bool
            傾きを計算する

        Returns
        -------
        """
        inputs = self.inputs
        values = self.values

        #　データがない場合は事前分布から予測を行う
        if inputs is None:
            return self.predict_from_prior(pred, full_cov, compute_grad)

        # 新規入力の次元が異なる場合はエラー
        if pred.ndim == 1:
            pred = np.array([pred])

        if pred.shape[1] != self.num_dims:
            raise Exception("Dimensionality of inputs must match dimensionality given at init time.")

        # 過去入力と新規入力間のカーネル関数を計算する
        cand_cross = self.noiseless_kernel.cross_cov(inputs, pred)
        # self.state(最新の状態)でのコレスキー分解の結果を取得する
        chol, alpha = self._pull_from_cache_or_compute()
        # 予測分散を計算するためのコレスキー分解
        beta = spla.solve_triangular(chol, cand_cross, lower=True)

        # 予測平均を計算
        func_m = np.dot(cand_cross.T, alpha) + self.mean.value

        # 予測分散を計算
        if full_cov:
            cand_cov = self.noiseless_kernel.cov(pred)
            func_v = cand_cov - np.sum(beta**2, axis=0)
        else:
            cand_cov = self.noiseless_kernel.diag_cov(pred)
            func_v = cand_cov - np.sum(beta**2, axis=0)

        if not compute_grad:
            return func_m, func_v

        grad_cross = self.noiseless_kernel.cross_cov_grad_data(inputs, pred)
        grad_xp_m = np.tensordot(np.transpose(grad_cross, (1,2,0)), alpha, 1)

        # this should be faster than (and equivalent to) spla.cho_solve((chol, True),cand_cross))
        gamma = spla.solve_triangular(chol.T, beta, lower=False)

        # Using sum and multiplication and summing instead of matrix multiplication
        # because I only want the diagonals of the gradient of the covariance matrix, not the whole thing
        grad_xp_v = -2.0*np.sum(gamma[:,:,np.newaxis] * grad_cross, axis=0)

        # Not very important -- just to make sure grad_xp_v.shape = grad_xp_m.shape
        if values.ndim > 1:
            grad_xp_v - grad_xp_v[:,:,np.newaxis]

        # In case this is a function over a 1D input,
        # return a numpy array rather than a float
        if np.ndim(grad_xp_m) == 0:
            grad_xp_m = np.array([grad_xp_m])
            grad_xp_v = np.array([grad_xp_v])

        return func_m, func_v, grad_xp_m, grad_xp_v

    # データがない場合に事前分布を用いて予測を行う
    def predict_from_prior(self, pred, full_cov=False, compute_grad=False):
        mean = self.mean.value * np.ones(pred.shape[0])
        if full_cov:
            cov = self.noiseless_kernel.cov(pred)
            return mean, cov
        elif compute_grad:
            var = self.noiseless_kernel.diag_cov(pred)
            grad = np.zeros((pred.shape[0], self.num_dims))
            return mean, var, grad, grad
        else:
            var = self.noiseless_kernel.diag_cov(pred)
            return mean, var


    ## 以下はサンプリング ##

    def sample_from_prior_given_hypers(self, pred, n_samples=1, joint=True):
        """ 指定されたハイパーパラメータの下で
            予測点において事前分布からサンプリングを行う
            i.e. y from p(y|theta)
        """
        N_pred = pred.shape[0]
        if joint: # 共分散考慮してサンプリング
            mean = self.mean.value
            cov  = self.noiseless_kernel.cov(pred)
            return npr.multivariate_normal(mean*np.ones(N_pred), cov, size=n_samples).T.squeeze()
        else:     # 独立にサンプリング
            mean = self.mean.value
            var  = self.noiseless_kernel.diag_cov(pred)
            return np.squeeze(mean + npr.randn(N_pred, n_samples) * np.sqrt(var)[:, None])

    def sample_from_prior(self, pred, n_samples=1, joint=True):
        """ ハイパーパラメータの事前分布の下で
            予測点において事前分布からサンプリングを行う
            i.e. y from p(y)
        """
        fants = np_zeros((pred.shape[0], n_samples))
        for i in range(n_samples):
            # ハイパーパラメータの事前分布からサンプリング
            for param in self.params:
                param.sample_from_prior()
            # 得られたサンプルの下で事前分布からサンプリング
            fants[:,i] = self.sample_from_prior_given_hypers(pred, joints)
        return fants.squeeze()


    def sample_from_posterior_given_hypers_and_data(self, pred, n_samples=1, joint=True):
        """ 指定されたハイパーパラメータの下で
            予測点において事後分布からサンプリングを行う
            i.e. y from p(y|theta,data)
        """
        if joint:
            predicted_mean, cov = self.predict(pred, full_cov=True)
            return npr.multivariate_normal(predicted_mean, cov, size=n_samples)
        else:
            predicted_mean, cov = self.predict(pred, full_cov=False)
            return np.squeeze(predicted_mean[:, None] + npr.randn(pred.shape[0], n_samples) * np.sqrt(var)[:,None])

######### 下記未実装 ########
#     def sample_from_posterior_given_data(self, pred, n_samples=1, joint=True):
#        """ ハイパーパラメータの事後分布の下で
#            予測点において事後分布からサンプリングを行う
#            i.e. y from p(y|theta,data)
#        """
#        fants = np.zeros((pred.shape[0], n_samples))
#        for i in range(n_samples):
#            self.generate_sample(1)
#            fants[:, i] = self.sample_from_posterior_given_hypers_sand_data(pred, joint)
#        return fants.squeeze()

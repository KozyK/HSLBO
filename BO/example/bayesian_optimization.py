# -*- coding: utf-8 -*-
# bayesian_optimization.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

# TODO: 入力の正規化

import numpy as np
import scipy.stats as sps

from pyDOE import lhs

# from .sampling.sobol import sobol_sequence
from .acquisition.expected_improvement import ExpectedImprovement

class BayesianOptimization():
    # acquisitionとtaskを受け取って最適化を進める．
    # 途中の進捗はすべてtaskに格納できるとわかりやすいかも

    def __init__(self, task, inputs=None, values=None, **options):
        # 解くべきタスク
        self.task = task

        # 探索空間の下界
        self.lower = self.task.lower

        # 探索空間の上界
        self.upper = self.task.upper

        # 発生させる初期値の数
        self.num_init = int(options.get("num_init", 1))

        self.acquisition = ExpectedImprovement(self.num_dims, self.lower, self.upper, num_manipulate=self.num_manipulate)

        self._initialize(inputs,values)

    def _initialize(self, inputs=None, values=None):
        # もし過去のデータがあれば格納する
        if (not inputs is None) and (not values is None):
            print("nanimonai")
            self.task.add_data(inputs,values)
        # なければランダムに初期実験計画を設定し評価
        else:
            # 初期実験計画をself.num_init回決める
            #init_inputs = sobol_sequence(self.num_manipulate, self.num_init, loc=self.lower, scale=(self.upper-self.lower))
            init_inputs = lhs(self.task.num_dims, samples=self.num_init, criterion='center') * (self.upper-self.lower) + self.lower
            # 初期実験計画を評価して格納
            self.task.evaluate(init_inputs)

    # 指定した回数だけ探索する
    def sequential_optimize(self,itertimes=1):
        # itertimesだけ探索を実行
        for i in range(itertimes):

            # 代理モデルを更新する
            self.acquisition.fit(self.task.inputs, self.task.values)

            # 現時点での最適値を取得
            current_best = self.current_best[1]

            # 獲得関数を最大化するself.next_searchを求める
            self.next_search = self.acquisition.maximize(current_best)

            # next_searchにおける関数値を評価してself.taskを更新
            self.task.evaluate(self.next_search)


    def next_search(self):
        self.acquisition.fit(self.task.inputs, self.task_values)

        return self.next_search

    # 外部からvalueを与える．
    def set_evaluation(self, value):
        self.task.add_data(self.next_search,value)
        print("The {0}th evaluation is added.".format(num_samples))


    def suggest(self, num): # num回の実験を終えた後の最適値を返す
        return self.task.suggest(num)

    @property
    def current_best(self): #現在の最適値
        return self.suggest(self.num_samples)

    @property
    def num_dims(self):
        return self.task.num_dims

    @property
    def num_manipulate(self):
        return self.task.num_manipulate

    @property
    def num_samples(self):
        return self.task.num_samples

    @property
    def inputs(self):
        return self.task.inputs

    @property
    def inputs_manipulate(self):
        return self.task.inputs_manipulate

    @property
    def inputs_conditions(self):
        return self.task.input_conditions

    @property
    def values(self):
        return self.task.values

if __name__ == '__main__':
    task = SimpleTask()
    bo = BayesianOptimization(task, inputs, values)
    bo.sequential_optimize(10)
    x, fun =bo.suggest(10)

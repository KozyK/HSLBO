# -*- coding: utf-8 -*-
# abstract_task.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

import numpy as np
import scipy.optimize as spo

from abc import ABCMeta, abstractmethod

class AbstractTask():
    __metaclass__ = ABCMeta

    @property
    def inputs(self):
        return None

    @property
    def values(self):
        return None

    @abstractmethod
    def objective(self, input):
        # inputに対して評価関数の値を返す関数
        pass

    @abstractmethod
    def evaluate(self, inputs):
        # 実際に関数の値を計算し，データを保持する
        pass

    @abstractmethod
    def add_data(self, inputs, values):
        # 過去のデータを格納する
        pass


class SimpleTask(AbstractTask):
    def __init__(self):
        self.num_manipulate = 2# 操作変数の数
        self.num_condition = 0# 条件となる変数の数(raw material attributeの次元数)

        self._upper = np.array([10,15]) # 上界
        self._lower = np.array([-5,0]) # 下界

        self._inputs = np.empty((0,self.num_dims)) #入力のデータ
        self._values =  np.empty((0,1))# 出力のデータ

    @property
    def num_dims(self):
        return self.num_manipulate+self.num_condition

    @property
    def num_samples(self):
        return self._inputs.shape[0]

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def inputs(self):
        return self._inputs

    @property
    def inputs_manipulate(self):
        return self._inputs[:,:self.num_manipulate]

    @property
    def inputs_conditions(self):
        return self._inputs[:,self.num_manipulate:]

    @property
    def values(self):
        return self._values

    # 条件変数を発生させる
#    def condition(self):
#        return None

    # 入力の各次元に独立なsin関数の和で表されるような評価関数
    def objective(self, x):
        return branin(x)

    # 関数評価してデータに追加
    def evaluate(self, inputs):
        # データが1次元なら2次元に整形
        if inputs.ndim ==1:
            inputs = np.array([inputs])

        for i in range(inputs.shape[0]):
            y = self.objective(inputs[i,:])

            self.add_data(inputs[i,:], y)

    # データを格納する
    def add_data(self, x, value):
        # もし上界と下界を超えていたらエラーを吐く

        if np.any(x < self._lower) or np.any(self._upper < x):
            raise NameError('Input is not in valid boundary')
        self._inputs = np.vstack((self._inputs, x))
        self._values = np.vstack((self._values, value))

    # num回の実験を終えた後の最適値を返す
    def suggest(self,num):
        idx_min = np.argmin(self._values[:num,:],axis=0)
        return self._inputs[idx_min], self._values[idx_min]
        # 2つめの返り値は2次元配列であることに注意

def branin(x):
    a = 1
    b = 5.1 / (4 * np.square(np.pi))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/ (8 * np.pi)

    f = a * np.square((x[1] - b * np.square(x[0]) + c * x[0] -r)) + s * (1-t) * np.cos(x[0]) + s
    return f

def branin_modified(x):
    a = 1
    b = 5.1 / (4 * np.square(np.pi))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/ (8 * np.pi)

    f = a * np.square((x[1] - b * np.square(x[0]) + c * x[0] -r)) + s * (1-t) * np.cos(x[0]) + s
    return f


if __name__ == '__main__':
    st = SimpleTask()
    for i in range(17):
        search = np.array([[0.0625*i,0.0625*i]])
        st.evaluate(search)

    for j in range(17):
        _, max = st.suggest(j)
        print(max)

# -*- coding: utf-8 -*-
# test2.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

def func1(self, **option1):
    a = option1.get("a", None)
    func2(a)

def func2(*option2):
    if option2[0]:
        say = option2[0].values
    else:
        say = None
    print(say)

if __name__ == '__main__':
    # func2に引数として通すための辞書
    aa = {"say" : 'Hello, world.'}

    func1('Hello world.', a=aa)

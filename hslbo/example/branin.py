# -*- coding: utf-8 -*-
# branin.py
#
# Coded by: Koji KITANO <koji.kitano52@gmail.com>,
# Human System Laboratory, Department of Systems Science,
# Graduate School of Informatics, Kyoto University.

import numpy as np
import numpy.random as npr

def branin(x):
    """ Branin function

    Input Domain
    ------------
        x1 : [-5, 10]
        x2 : [ 0, 15]

    Global Minimum
    --------------
        x = (-pi, 12.275), (pi, 2.275), (9.424478, 2.475)
        f = 0.397887

    """
    a = 1
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/ (8 * np.pi)

    term1 = a * (x[1] - b * (x[0]**2) + c * x[0] - r)**2
    term2 = s * (1-t) * np.cos(x[0])

    f = term1 + term2 + s

    return f

def branin_modified(x):
    """ Modified Branin function

    Input Domain
    ------------
        x1 : [-5, 10]
        x2 : [ 0, 15]

    Global Minimum
    --------------
        x = (-pi, 12.275)
        f = -15.31007

    Note
    ----
    It is modified so that there are two local minima and only one golbal minima, making it more representitive of engineering functions.

    """
    a = 1
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/ (8 * np.pi)

    term1 = a * (x[1] - b * (x[0]**2) + c * x[0] - r)**2
    term2 = s * (1-t) * np.cos(x[0])

    f = term1 + term2 + s + 5 * x[0]

    return f

def branin_modified_with_noise(x):
    """ Modified Branin function with gaussian noise

    Input Domain
    ------------
        x1 : [-5, 10]
        x2 : [ 0, 15]

    Global Minimum
    --------------
        x = (-pi, 12.275)
        f = -15.31007

    Observation noise
    -----------------
    ε ~ N(0, (0.1)^2)
    Note: It can be also interpleted as disturbance.


    Note
    ----
    It is modified so that there are two local minima and only one golbal minima, making it more representitive of engineering functions.

    """
    a = 1
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/ (8 * np.pi)

    term1 = a * (x[1] - b * (x[0]**2) + c * x[0] - r)**2
    term2 = s * (1-t) * np.cos(x[0])

    f = term1 + term2 + s + 5 * x[0] + npr.normal(0, 0.1)

    return f



if __name__ == '__main__':
    import numpy as np
    a = branin((-np.pi, 12.275))
    b = branin_modified((-np.pi, 12.275))
    print("a={0}".format(a))
    print("b={0}".format(b))

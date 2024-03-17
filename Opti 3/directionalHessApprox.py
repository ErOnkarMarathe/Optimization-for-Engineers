# Optimization for Engineers - Dr.Johannes Hild
# Directional Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: Hessian times direction, column vector in R ** n

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])

# dH = directionalHessApprox(myObjective, x, d)
# should return dH = [[1.55491],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23146448
    return matrnr


def directionalHessApprox(f, x: np.array, d: np.array, delta=1.0e-6, verbose=0):
    if verbose:
        print('Start directionalHessApprox...')

    norm_d = np.linalg.norm(d)

    a = x + ((delta / norm_d) * d)
    b = x - ((delta / norm_d) * d)

    grada = f.gradient(a)
    gradb = f.gradient(b)

    dH = (norm_d / (2 * delta)) * (grada - gradb)

    if verbose:
        print('directionalHessApprox terminated with dH=', dH)

    return dH







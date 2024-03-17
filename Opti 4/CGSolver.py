# Optimization for Engineers - Dr.Johannes Hild
# Conjugate Gradient Solver

# Purpose: CGSolver finds y such that norm(A * y - b) <= delta

# Input Definition:
# A: real valued spd matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n (solution in domain space)

# Required files:
# <none>


# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = CGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = CGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23146448
    return matrnr


def CGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start CGSolver...')

    countIter = 0

    x = b
    r = A @ x - b
    d = -r.copy()
    delta_new = np.dot(r.T, r)

    while np.linalg.norm(r) > delta:
        q = A @ d
        alpha = delta_new / np.dot(d.T, q)
        x = x + alpha * d
        r = r + alpha * q
        delta_old = delta_new
        delta_new = np.dot(r.T, r)
        beta = delta_new / delta_old
        d = -r + beta * d
        countIter += 1

    if verbose:
        print('CGSolver terminated after', countIter, 'steps with norm of residual being', np.linalg.norm(r))

    return x


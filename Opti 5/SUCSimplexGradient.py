# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = nonlinearObjective()
# x = np.array([[-0.015793], [0.012647]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSimplexGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0]]

# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSimplexGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23146448
    return matrnr


def SUCSimplexGradient(f, x: np.array, h, verbose=0):
    if verbose:
        print('Start SUCSimplexGradient...')

    grad_f_h = np.zeros_like(x)
    n = len(x)

    for j in range(n):
        x_j_plus = x.copy()
        x_j_minus = x.copy()

        x_j_plus[j] += h
        x_j_minus[j] -= h

        delta_f = f.objective(x_j_plus) - f.objective(x_j_minus)
        grad_f_h[j] = delta_f / (2 * h)

    if verbose:
        print('SUCSimplexGradient terminated with gradient =', grad_f_h)

    return grad_f_h


def SUCStencilFailure(f, x: np.array, h, verbose=0):
    if verbose:
        print('Check for SUCStencilFailure...')

    stenFail = False
    n = len(x)

    for j in range(n):
        x_j_plus = x.copy()
        x_j_minus = x.copy()

        x_j_plus[j] += h
        x_j_minus[j] -= h

        f_xstar = f.objective(x)
        f_x_j_plus = f.objective(x_j_plus)
        f_x_j_minus = f.objective(x_j_minus)

        if f_xstar >= f_x_j_plus or f_xstar <= f_x_j_minus:  # Modified condition
            stenFail = True
            break

    if verbose:
        print('SUCStencilFailure check returns', stenFail)

    return stenFail





# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[-0.6], [0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]

import numpy as np
from numpy import linalg
import WolfePowellSearch as WP
import directionalHessApprox as DHA


def matrnr():
    # set your matriculation number here
    matrnr = 23146448
    return matrnr

def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise ValueError('Invalid value for eps')

    if verbose:
        print('Start inexactNewtonCG...')

    countIter = 0
    x = x0.copy()
    gradx = f.gradient(x)
    eta = np.minimum(1 / 2, (np.sqrt(np.linalg.norm(f.gradient(x))) * np.linalg.norm(f.gradient(x))))

    while np.linalg.norm(f.gradient(x)) > eps:
        dk = - (f.gradient(x))
        dH = DHA.directionalHessApprox(f, x, dk)  # Call the directionalHessApprox function using DHA as the abbreviation
        pk = np.transpose(dk) @ dH
        if pk > eps * np.square(np.linalg.norm(dk)):
            rj = f.gradient(x)
            dj = -rj
            xj = x.copy()
            dA = DHA.directionalHessApprox(f, x, dk)
            pj = pk.copy()

            tj = np.square(np.linalg.norm(rj)) / pj
            xj = xj + (tj * dj)
            rold = rj.copy()
            rj = rold + (tj * dA)
            betaj = np.square(np.linalg.norm(rj)) / np.square(np.linalg.norm(rold))
            dj = -rj + (betaj * dj)

            while np.linalg.norm(rj) > eta:
                dA = DHA.directionalHessApprox(f, x, dj)  # Call the directionalHessApprox function using DHA as the abbreviation
                pj = np.transpose(dj) @ dA
                tj = np.square(np.linalg.norm(rj)) / pj
                xj = xj + (tj * dj)
                rold = rj.copy()
                rj =rold + (tj * dA)
                betaj = np.square(np.linalg.norm(rj)) / np.square(np.linalg.norm(rold))
                dj = -rj + (betaj * dj)

            dk = xj - x
        alpha = WP.WolfePowellSearch(f, x, dk)# Calculate step size tk for f at x in direction dk using WolfePowell
        x = x + (alpha * dk)
        eta = np.minimum(1 / 2, (np.sqrt(np.linalg.norm(f.gradient(x))) * np.linalg.norm(f.gradient(x))))

        countIter =countIter + 1

    if verbose:
        gradx = f.gradient(x)
        print('inexactNewtonCG terminated after', countIter, 'steps with norm of gradient =', np.linalg.norm(gradx))

    return x






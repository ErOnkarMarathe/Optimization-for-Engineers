# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0.1], [0.1]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[1.5], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[-0.26],[0.21]] (if it is close to [[0.26],[-0.21]] then maybe your reduction is done wrongly)

# myObjective = bananaValleyObjective()
# a = np.array([[-10], [-10]])
# b = np.array([[10], [10]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0], [1]], dtype=float)
# eps = 1.0e-6
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]] in less than 30 iterations. If you have too much iterations, then maybe the hessian is used wrongly.


import numpy as np
import projectedBacktrackingSearch as PB


def matrnr():
    # set your matriculation number here
    matrnr = 23146448
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0, sigma=1.0e-4):
    if eps <= 0:
        raise TypeError('Range of eps is wrong!')

    if verbose:
        print('Start projectedBFGSDescent...')

    countIter = 0
    xk = P.project(x0)
    gradx = f.gradient(xk)
    Bk = np.eye(len(xk))  # Initialize Bk as the identity matrix
    while np.linalg.norm(xk - P.project(xk - f.gradient(xk))) > eps:
        # Step (a): Reduce Bk depending on the active set A(x)
        active_indices = P.activeIndexSet(xk)
        Bk[active_indices, :] = Bk[active_indices, :]
        Bk[:, active_indices] = Bk[:, active_indices]

        # Step (b): Check if dk is a descent direction
        dk = -Bk @ f.gradient(xk)
        if f.gradient(xk).T @ dk >= 0:
            dk = -gradx  # Switch to steepest descent
            Bk = np.eye(len(xk))  # Reset Bk to the identity matrix

        # Step (c): Find tk using line search with different arguments
        tk = PB.projectedBacktrackingSearch(f, P, xk, dk)  # Modified line search

        # Step (d): Update xk and Bk
        xk_old = xk
        xk = P.project(xk + tk * dk)
        sk = xk - xk_old  # sk = change in x
        yk = f.gradient(xk) - f.gradient(xk_old)  # yk = change in gradient

        rho_k = sk -Bk.T @ yk
        Bk = Bk + ((rho_k @ sk.T + sk @ rho_k.T)/(yk.T @ sk)) - (((rho_k.T @ yk) / ((yk.T @ sk)**2)) * (sk @ sk.T))

        countIter += 1

    if verbose:
        print('projectedBFGSDescent terminated after', countIter, 'steps with norm of stationarity =',
              np.linalg.norm(xk - P.project(xk - gradx)))

    return xk





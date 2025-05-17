from collections import namedtuple
import numpy as np

MinimizeScalarResult = namedtuple("MinimizeScalarResult", ["root", "f_min", "converged", "iterations"])


def minimize_scalar_positive(f, bracket, xtol=1e-5, max_iter=100):
    """
    Minimize a scalar function using the golden section search method.
    This method is suitable for functions that are unimodal in the given interval.
    Parameters:
        f: The function to minimize. It should take a single argument and return a scalar.
        bracket: A tuple (a, b) defining the interval in which to search for the minimum.
        xtol: The tolerance for convergence. The search stops when the interval size is less than xtol.
        max_iter: The maximum number of iterations to perform.
    Returns:
        MinimizeScalarResult: A named tuple containing the following fields:
            - root: The x value at which the minimum occurs.
            - f_min: The minimum value of the function.
            - converged: A boolean indicating whether the algorithm converged.
            - iterations: The number of iterations performed.
    """
    a, b = bracket
    gr = (np.sqrt(5) + 1) / 2  # golden ratio

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    fc, fd = f(c), f(d)

    for i in range(max_iter):
        if abs(b - a) < xtol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)

    x_min = (b + a) / 2
    return MinimizeScalarResult(root=x_min, f_min=f(x_min), converged=(abs(b - a) < xtol), iterations=i+1)

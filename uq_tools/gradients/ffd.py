import numpy as np


def approximateGradientAt(x, f, hs=1e-3):
    """
    Approximates gradient by forward finite differences at a specified location.

    :param x: Location
    :param f: Function
    :param float h: Discretization parameter
    """
    dimIn = len(x)

    fx = f(x)

    xhs = np.multiply(np.eye(dimIn), hs) + x
    fxhs = [f(xh) for xh in xhs]

    derivs = np.divide(np.transpose(fxhs) - fx, hs).T

    return derivs if np.isscalar(derivs[0]) else derivs.T[0]

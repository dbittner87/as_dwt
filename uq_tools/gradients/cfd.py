import numpy as np


def approximateGradientAt(x, f, h=1e-3):
    """
    Approximates gradient by central finite differences at a specified location.

    :param x: Location
    :param f: Function
    :param float h: Discretization parameter
    """
    dimIn = len(x)

    f_diffs = [f(x + h_) - f(x - h_) for h_ in np.multiply(np.eye(dimIn), h)]

    derivs = np.divide(f_diffs, 2 * h)

    return derivs if np.isscalar(derivs[0]) else derivs.T

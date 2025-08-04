import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline
from math import pi, cos
from typing import List

def lagrangeBase(n: int, k: int, points: List[Tuple[float, float]]) -> np.poly1d:
    """
    This function calculates the Lagrange base polynomial.

    Parameters:
    n (int): The number of points.
    k (int): The index of the point.
    points (List[Tuple[float, float]]): The list of points.

    Returns:
    np.poly1d: The Lagrange base polynomial.
    """
    x = np.array([point[0] for point in points])
    xk = x[k]
    x = np.delete(x, k)
    poly_x = np.poly(x)
    polyval_xk = np.polyval(poly_x, xk)
    return np.poly1d(poly_x / polyval_xk if polyval_xk != 0 else poly_x)

def lagrangePol(points: List[Tuple[float, float]]) -> np.poly1d:
    """
    This function calculates the Lagrange polynomial.

    Parameters:
    points (List[Tuple[float, float]]): The list of points.

    Returns:
    np.poly1d: The Lagrange polynomial.
    """
    n = len(points)
    pol = np.poly1d([0])
    for k in range(n):
        pol = np.polyadd(pol, points[k][1] * lagrangeBase(n, k, points))
    return pol


def cubic_spline_interpolation(points: List[Tuple[float, float]]) -> CubicSpline:
    """
    This function calculates the interpolation with Cubic Splines.

    Parameters:
    points (List[Tuple[float, float]]): Points list.

    Returns:
    CubicSpline: Cubic Spline.
    """
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    cs = CubicSpline(x, y)
    return cs

def chebyshevNodes(n: int, a: float = -1, b: float = 1) -> np.ndarray:
    """
    This function calculates the Chebyshev nodes.

    Parameters:
    n (int): The number of nodes.
    a (float): The lower limit of the interval. Default is -1.
    b (float): The upper limit of the interval. Default is 1.

    Returns:
    np.ndarray: The array of Chebyshev nodes.
    """
    return np.array([0.5*(a+b) + 0.5*(b-a)*cos((2*k-1)*pi/(2*n)) for k in range(1, n+1)])


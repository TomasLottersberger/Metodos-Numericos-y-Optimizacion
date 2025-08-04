import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from methods import cubic_spline_interpolation, chebyshevNodes, lagrangePol
from typing import Callable
from errors import evaluate_difference

def plotGraph(ax: Any, x: np.ndarray, y: np.ndarray, label: str, linestyle: str, marker: str) -> None:
    ax.plot(x, y, label=label, linestyle=linestyle, marker=marker)

def plotFunction(func: Any, x, ax) -> None:
    y_func = func(x)
    plotGraph(ax, x, y_func, 'Ground Truth', 'solid', None)

def plotCubicSplineEqui(func, x, ax,nodos) -> None:
    x_equi= np.linspace(-1, 1, nodos)
    points1= np.array([[x_equi[i], func(x_equi[i])] for i in range(nodos)])
    cs1 = cubic_spline_interpolation(points1)
    y1 = cs1(x)
    plotGraph(ax, x, y1, f'{nodos} nodos equispaciados', '--', None)

def plotCubicSplineChe(func, x, ax, nodos) -> None:
    x_cheb = chebyshevNodes(nodos)
    points2 = np.array([[x_cheb[i], func(x_cheb[i])] for i in range(nodos)])
    points2 = points2[::-1]
    cs1 = cubic_spline_interpolation(points2)
    y1 = cs1(x_cheb)
    plotGraph(ax, x_cheb, y1, f'{nodos} nodos por Chebyshev', ':', None)
    
def plotLEqui(func, x, ax,nodos) -> None:
    x_equi= np.linspace(-1, 1, nodos)
    points1: np.ndarray = np.array([[x_equi[i], func(x_equi[i])] for i in range(nodos)])
    pol1: Callable[[np.ndarray], np.ndarray] = lagrangePol(points1)
    y1: np.ndarray = pol1(x)
    plotGraph(ax, x, y1, f'{nodos} nodos equispaciados', '--', None)

def plotLChebyshev(func, x, ax,nodos) -> None:
    x_cheb: np.ndarray = chebyshevNodes(nodos)
    points2: np.ndarray = np.array([[x_cheb[i], func(x_cheb[i])] for i in range(nodos)])
    pol2: Callable[[np.ndarray], np.ndarray] = lagrangePol(points2)
    y2: np.ndarray = pol2(x)
    plotGraph(ax, x, y2, f'{nodos} nodos por Chebyshev', ':', None)

def plotError(func, pol, interval, distance):
    point_list = evaluate_difference(func, pol, interval, distance)
    x = [elem[0] for elem in point_list]
    y = [elem[1] for elem in point_list]
    plt.plot(x, y, label='Error', linestyle='solid', marker=None)

def plotErrorsEquispacedPoints(func, interval, max_points):
    data = []
    for n in range(2, max_points + 1):
        x = np.linspace(interval[0], interval[1], n)
        y = func(x)
        points = np.array(list(zip(x, y)))
        pol = lagrangePol(points)
        point_list = evaluate_difference(func, pol, interval)
        data.append([elem[1] for elem in point_list])
    
def plotErrorsNonEquispacedPoints(func, interval, max_points):
    data = []
    for n in range(2, max_points + 1):
        x_cheb = chebyshevNodes(n)
        y = func(x_cheb)
        points = np.array(list(zip(x_cheb, y)))
        pol = lagrangePol(points)
        point_list = evaluate_difference(func, pol, interval)
        data.append([elem[1] for elem in point_list])
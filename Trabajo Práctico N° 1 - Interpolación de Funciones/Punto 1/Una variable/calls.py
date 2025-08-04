import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from methods import cubic_spline_interpolation, chebyshevNodes, lagrangePol
from errors import median_error, max_relative_error
from graphs import plotErrorsEquispacedPoints, plotErrorsNonEquispacedPoints, plotFunction, plotCubicSplineEqui, plotCubicSplineChe, plotLEqui, plotLChebyshev, plotError
from tabulate import tabulate

def main():
    func = lambda x: (-1)*(0.4)*np.tanh(50*x)+(0.6)
    interval = (-1, 1)
    max_points = 21
    x = np.linspace(interval[0], interval[1], 1000)
    fig1, ax1 = plt.subplots()
    plotFunction(func, x, ax1)
    plotLEqui(func, x, ax1, 10)
    plotLChebyshev(func, x, ax1, 10)
    ax1.set_title('Interpolación de Lagrange con 10 nodos')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    plt.show()
    x = np.linspace(interval[0], interval[1], 1000)
    fig1, ax1 = plt.subplots()
    plotFunction(func, x, ax1)
    plotCubicSplineEqui(func, x,ax1,10)
    plotCubicSplineChe(func, x,ax1,10)
    ax1.set_title('Interpolación de Splines Cúbicos con 10 nodos')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    plt.show()
    fig7, ax7 = plt.subplots()
    x_equi = np.linspace(interval[0], interval[1], 10)
    points_equi = np.array([[x_equi[i], func(x_equi[i])] for i in range(10)])
    pol_equi = lagrangePol(points_equi)
    plotError(func, pol_equi, interval, 1000)
    x_cheb = chebyshevNodes(10)
    points_cheb = np.array([[x_cheb[i], func(x_cheb[i])] for i in range(10)])
    pol_cheb = lagrangePol(points_cheb)
    plotError(func, pol_cheb, interval, 1000)
    ax7.set_title('Comparación de Error: Equispaciados vs Chebyshev')
    ax7.set_xlabel('x')
    ax7.set_ylabel('Error')
    ax7.legend(['Nodos Equispaciados', 'Nodos Chebyshev'])
    plt.show()


if __name__ == "__main__":
    main()
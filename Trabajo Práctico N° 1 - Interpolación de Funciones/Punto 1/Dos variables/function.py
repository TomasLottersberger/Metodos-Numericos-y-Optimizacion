import numpy as np
from math import pi, cos

def fb(x):
    term1 = 0.75 * np.exp(-((9*x[0] - 2)**2)/4 - ((9*x[1] - 2)**2)/4)
    term2 = 0.75 * np.exp(-((9*x[0] + 1)**2)/49 - ((9*x[1] + 1)**2)/10)
    term3 = 0.5 * np.exp(-((9*x[0] - 7)**2)/4 - ((9*x[1] - 3)**2)/4)
    term4 = -0.2 * np.exp(-((9*x[0] - 7)**2)/4 - ((9*x[1] - 3)**2)/4)
    
    return term1 + term2 + term3 + term4

def chebyshevNodes(n: int, a: float = -1, b: float = 1) -> np.ndarray:
    return np.array([0.5*(a+b) + 0.5*(b-a)*cos((2*k-1)*pi/(2*n)) for k in range(1, n+1)])
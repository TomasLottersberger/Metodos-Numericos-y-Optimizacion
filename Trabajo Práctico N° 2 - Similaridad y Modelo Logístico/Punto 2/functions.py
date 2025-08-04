import numpy as np

def logistic_model(N, r, K, A):
    return r * N * (K - N) / K

def model_variant(N, r, K, A):
    return r * N * (1 - N/K) * (N/A - 1)

def analytic_sol(r, K, n0, t):
    return K * n0 * np.exp(r*t) / (K + n0 * (np.exp(r*t) - 1))

def ground_truth(r, K, n0, t_max, dt):
    t = np.arange(0, t_max, dt)
    N = analytic_sol(r, K, n0, t)
    return t, N

def euler_method(N0, r, K, A, dt, t_max, func):
    t = np.arange(0, t_max, dt)
    N = np.zeros_like(t)
    N[0] = N0
    for i in range(1, len(t)):
        N[i] = N[i-1] + func(N[i-1], r, K, A) * dt
    return t, N

def runge_kutta_method(N0, r, K, A, dt, t_max, func):
    t = np.arange(0, t_max, dt)
    N = np.zeros_like(t)
    N[0] = N0
    for i in range(1, len(t)):
        k1 = func(N[i-1], r, K, A)
        k2 = func(N[i-1] + 0.5 * k1 * dt, r, K, A)
        k3 = func(N[i-1] + 0.5 * k2 * dt, r, K, A)
        k4 = func(N[i-1] + k3 * dt, r, K, A)
        N[i] = N[i-1] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return t, N
import numpy as np
import matplotlib.pyplot as plt
import functions

r = 0.1
K = 100
A = 10
N0 = 20
dt = 2
t_max = 45

def comparition_ground_truth_with_methods():
    t_analytical, N_analytical = functions.ground_truth(r, K, N0, t_max, dt)
    t_euler, N_euler = functions.euler_method(N0, r, K, A, dt, t_max, functions.logistic_model)
    t_rk, N_rk = functions.runge_kutta_method(N0, r, K, A, dt, t_max, functions.logistic_model)
    plt.figure(figsize=(10, 6))
    plt.plot(t_analytical, N_analytical[:50], label=f'Solución analítica', color='red')
    plt.plot(t_rk, N_rk, label=f'Método de Runge-Kutta', color='orange')
    plt.plot(t_euler, N_euler, label=f'Método de Euler', color='blue')
    plt.xlabel('Tiempo')
    plt.ylabel('Tamaño de la población')
    plt.legend()
    plt.show()

def comparition_different_dt_values():
    t_max = 25
    dt_values = [2, 1, 0.1]
    for dt in dt_values:
        t_euler, N_euler = functions.euler_method(N0, r, K, A, dt, t_max, functions.model_variant)
        t_rk, N_rk = functions.runge_kutta_method(N0, r, K, A, dt, t_max, functions.model_variant)
        plt.figure(figsize=(10, 6))
        plt.plot(t_euler, N_euler, label=f'Método de Euler (dt={dt})', color='blue')
        plt.plot(t_rk, N_rk, label=f'Método de Runge-Kutta (dt={dt})', color='orange')
        plt.axhline(K, color='green', linestyle='--', label=f'Capacidad de carga (K={K})')
        plt.axhline(A, color='red', linestyle='--', label=f'Umbral mínimo de supervivencia (A={A})')
        plt.xlabel('Tiempo')
        plt.ylabel('Tamaño de la población')
        plt.title(f'Comparación con dt={dt}')
        plt.legend()
        plt.show()

def comparition_different_initial_conditions():
    t_max = 10
    initial_conditions = [5, 50, 110]
    results = []
    for N0 in initial_conditions:
        t_rk, N_rk = functions.runge_kutta_method(N0, r, K, A, dt, t_max, functions.model_variant)
        results.append((t_rk, N_rk))
    plt.figure(figsize=(10, 6))
    plt.plot(results[0][0], results[0][1], label=f'Condición inicial N0=5', color='orange')
    plt.plot(results[1][0], results[1][1], label=f'Condición inicial N0=50', color='brown')
    plt.plot(results[2][0], results[2][1], label=f'Condición inicial N0=110', color='blue')
    plt.axhline(K, color='green', linestyle='--', label=f'Capacidad de carga (K={K})')
    plt.axhline(A, color='red', linestyle='--', label=f'Umbral mínimo de supervivencia (A={A})')
    plt.xlabel('Tiempo')
    plt.ylabel('Tamaño de la población')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #comparition_ground_truth_with_methods()
    comparition_different_dt_values()
    #comparition_different_initial_conditions()
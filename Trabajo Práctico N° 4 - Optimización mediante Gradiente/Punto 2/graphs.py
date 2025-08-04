import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import functions

def mse_for_pseudoinverse_graph(mse_pseudo_train, mse_pseudo_test):
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Test"], [mse_pseudo_train, mse_pseudo_test], color=['red', 'blue'])
    plt.title("MSE - Pseudoinversa")
    plt.ylabel("MSE")
    plt.ylim([0, max(mse_pseudo_train, mse_pseudo_test) * 1.2])
    plt.show()

def mse_for_gd_graph(mse_train_history, mse_test_history):
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Test"], [mse_train_history[-1], mse_test_history[-1]], color=['red', 'blue'], alpha=0.7)
    plt.title("MSE - Gradiente Descendiente")
    plt.ylabel("MSE")
    plt.ylim([0, max(mse_train_history[-1], mse_test_history[-1]) * 1.2])
    plt.show()

def mse_per_iteration_gd_graph(mse_train_history, mse_test_history):
    plt.figure(figsize=(8, 6))
    plt.plot(mse_train_history, label="Train Error", color='red')
    plt.plot(mse_test_history, label="Test Error", color='blue')
    plt.title("Evolución del error en gradiente descendiente")
    plt.xlabel("Iteraciones")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def solution_comparation_graph(w_pseudo, w_gd):
    plt.figure(figsize=(8, 6))
    indices = np.arange(len(w_pseudo))
    plt.plot(indices, w_pseudo, 'o-', label="Pseudoinversa", color='green')
    plt.plot(indices, w_gd, 'x-', label="Gradiente Descendiente", color='purple')
    plt.title("Comparación de soluciones (coeficientes w)")
    plt.xlabel("Índice de componente")
    plt.ylabel("Valor de componente")
    plt.legend()
    plt.show()

def learning_rates_comparation(X_train, y_train, eta_optima):
    etas = [eta_optima / 10, eta_optima, 10 * eta_optima]
    labels = [f"η pequeño = 3 * 10^(-6)", f"η optima = 3 * 10^(-5)", f"η grande = 3 * 10^(-4)"]
    plt.figure(figsize=(10, 6))
    for eta, label in zip(etas, labels):
        errors = functions.gradient_descent_analysis(X_train, y_train, eta)
        plt.plot(errors, label=label)
    plt.xlabel("Iteraciones")
    plt.ylabel("Error (MSE)")
    plt.title("Impacto de la tasa de aprendizaje en gradiente descendiente")
    plt.legend()
    plt.show()

def trajectory_gd_graph(trajectory, w_pseudo):
    trajectory = np.array(trajectory)
    num_coeffs_to_plot = min(1, trajectory.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(num_coeffs_to_plot):
        plt.plot(trajectory[:, i], label=f"w[{i}] (GD)", alpha=0.8)  # Gradiente descendente
        plt.axhline(y=w_pseudo[i], color="r", linestyle="--", label=f"w_pseudo[{i}] (Pseudoinversa)")
    plt.title("Convergencia de coeficientes representativos hacia la solución de la pseudoinversa")
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de los coeficientes")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.show()

def difference_graph(w_pseudo, w_gd):
    difference = w_gd - w_pseudo
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(difference)), difference, marker="o", linestyle="--", color="red", label="w_gd - w_pseudo")
    plt.axhline(0, color="blue", linestyle="-", linewidth=1, label="Diferencia = 0")
    plt.title("Diferencia entre w_gd y w_pseudo")
    plt.xlabel("Índice del coeficiente")
    plt.ylabel("Diferencia")
    plt.legend()
    plt.show()

def iteration_comparation_graph(w_gd, w_gd1, w_gd2, w_pseudo):
    plt.figure(figsize=(8, 6))
    indices = np.arange(len(w_gd))
    plt.plot(indices, w_pseudo, 'o-', label="Solución exacta", color='black')
    plt.plot(indices, w_gd1, 'x-', label="Solución con 1000 iteraciones", color='blue')
    plt.plot(indices, w_gd, 'x-', label="Solución con 10000 iteraciones", color='red')
    plt.plot(indices, w_gd2, 'x-', label="Solución con 30000 iteraciones", color='green')
    plt.title("Comparación de soluciones con diferente cantidad de iteraciones")
    plt.xlabel("Índice de componente")
    plt.ylabel("Valor de componente")
    plt.legend()
    plt.show()
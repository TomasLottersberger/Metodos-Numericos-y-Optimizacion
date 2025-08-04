import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset sin la primera fila y columna (ambas son índices)
X = pd.read_csv("Punto 2\dataset_x_y\dataset01.csv", header=0, index_col=0).values

# Cargar etiquetas Y
y = np.loadtxt("Punto 2/dataset_x_y/y1.txt")

# Graficar la matriz de datos completa con énfasis en las últimas 6 filas y columnas
plt.figure(figsize=(10, 10))
sns.heatmap(X, cmap="viridis", cbar=True)
plt.title("Matriz de Datos Original")

# Resaltar las últimas 6 filas y columnas
plt.axhline(y=X.shape[0] - 6, color="r", linestyle="--", linewidth=2)
plt.axvline(x=X.shape[1] - 6, color="r", linestyle="--", linewidth=2)
plt.show()

# Centrar los datos en X e y
X_centered = X - np.mean(X, axis=0)
y_centered = y - np.mean(y)  # Restar la media de Y

# 1. Aplicar PCA a toda la matriz X_centered
XTX = np.dot(X_centered.T, X_centered)
eigenvalues, eigenvectors = np.linalg.eig(XTX)
sorted_indices = np.argsort(eigenvalues)[::-1]

# Función para reducir la dimensionalidad con PCA
def reduce_dimension_with_pca(X, eigenvectors, d):
    Vd = eigenvectors[:, sorted_indices[:d]]  # Seleccionar los primeros d vectores propios
    Z = np.dot(X, Vd)  # Proyectar al espacio reducido
    return Z

# Reducir la dimensionalidad para d = 2, 6, 10, y el máximo permitido
d_values = [2, 6, 10, X.shape[1]]
Z_pca = {}

for d in d_values:
    Z_pca[d] = reduce_dimension_with_pca(X_centered, eigenvectors, d)

# Calcular y visualizar las matrices de similaridad
sigma = np.std(X_centered)  # Parámetro sigma

# Definir función de similaridad
def similarity_matrix(X, sigma):
    n = X.shape[0]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(X[i] - X[j]) ** 2
            sim_matrix[i, j] = np.exp(-dist / (2 * sigma**2))
    return sim_matrix

# Similaridad en el espacio original
sim_X = similarity_matrix(X_centered, sigma)
sns.heatmap(sim_X, cmap="viridis")
plt.title("Matriz de Similaridad en el Espacio Original")
plt.show()

# Similaridad en espacios reducidos con PCA
for d in d_values:
    sim_Z = similarity_matrix(Z_pca[d], sigma)
    plt.figure()
    sns.heatmap(sim_Z, cmap="viridis")
    plt.title(f"Matriz de Similaridad en Espacio Reducido con PCA (d={d})")
    plt.show()

    # Calcular el error de Frobenius entre sim_X y sim_Z
    error_frobenius = np.linalg.norm(sim_X - sim_Z)
    print(f"Error de Frobenius para d={d}: {error_frobenius}")

# Ajuste del modelo de mínimos cuadrados en el espacio original
beta = np.linalg.pinv(X_centered) @ y_centered  # Usando y_centered
print("Vector de pesos β en el espacio original:", beta)

# Ajuste en el espacio reducido con d=2
Z_2 = Z_pca[2]
beta_Z2 = np.linalg.pinv(Z_2) @ y_centered  # Usando y_centered
y_pred_Z2 = Z_2 @ beta_Z2

# Predicción en el espacio original
y_pred_X = X_centered @ beta

# Calcular errores en el espacio original y en el espacio reducido (comparando con y)
n = len(y)
error_original = np.linalg.norm(y - y_pred_X) ** 2/ n
error_reducido = np.linalg.norm(y - y_pred_Z2) ** 2/ n

print(f"Error en el espacio original: {error_original}")
print(f"Error en el espacio reducido (d=2): {error_reducido}")

# Encontrar las muestras con mejor predicción en el espacio original
best_predictions_original = np.argsort(np.abs(y - y_pred_X))
print(
    "Índices de las muestras con mejor predicción (espacio original):",
    best_predictions_original,
)

# Encontrar las muestras con mejor predicción en el espacio reducido (d=2)
best_predictions_reduced = np.argsort(np.abs(y_centered - y_pred_Z2))
print(
    "Índices de las muestras con mejor predicción (espacio reducido, d=2):",
    best_predictions_reduced,
)
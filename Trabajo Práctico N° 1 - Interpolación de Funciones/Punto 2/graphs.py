import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory_3d(trajectory, interpolated_trajectory, estimated_positions):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], label='Trayectoria Real', color='blue')
    ax.plot(interpolated_trajectory[:, 0], interpolated_trajectory[:, 1], interpolated_trajectory[:, 2], 
        label='Trayectoria Estimada', color='red', linestyle='dashed')
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], 
           color='red', label='Posiciones Estimadas', s=50)
    #ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2], 
    #       label='Sensores', color='green', marker='o', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparación de Trayectorias en 3D')
    ax.legend()
    plt.show()

def plot_interpolation_error(errors, times):
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(times, errors['error_x'], color='red', label='Error en X')
    plt.xlabel('Tiempo')
    plt.ylabel('Error en X')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(times, errors['error_y'], color='green', label='Error en Y')
    plt.xlabel('Tiempo')
    plt.ylabel('Error en Y')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(times, errors['error_z'], color='blue', label='Error en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Error en Z')
    plt.legend()
    
    plt.suptitle('Errores de Interpolación en los Ejes X, Y, Z')
    plt.tight_layout()
    plt.show()
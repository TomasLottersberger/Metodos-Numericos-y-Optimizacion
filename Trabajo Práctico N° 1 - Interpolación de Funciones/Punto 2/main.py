import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
import graphs

def open_file(path):
    opened = []
    with open(path, 'r') as file:
        next(file)
        for line in file:
            values = line.strip().split(',')
            values = [float(v) for v in values]
            opened.append(values)
    return opened

def system_of_equations(variables, sensor_positions, distances):
    x, y, z = variables
    equations = []
    for i in range(3):
        xi, yi, zi = sensor_positions[i]
        di = distances[i]
        equation = np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - di
        equations.append(equation)
    return equations

def calculate_interpolation_error(real_trajectory, interpolated_trajectory):
    error_x = np.abs(real_trajectory[:, 0] - interpolated_trajectory[:, 0])
    error_y = np.abs(real_trajectory[:, 1] - interpolated_trajectory[:, 1])
    error_z = np.abs(real_trajectory[:, 2] - interpolated_trajectory[:, 2])
    return {'error_x': error_x, 'error_y': error_y, 'error_z': error_z}

def main():
    measurements = open_file('Punto 2\data\measurements.txt')
    measurements = np.array(measurements)
    measurements = np.array(measurements[:, 1:])
    sensor_positions = open_file('Punto 2\data\sensor_positions.txt')
    sensor_positions = np.array(sensor_positions)
    sensor_positions = np.array(sensor_positions[:, 1:])
    trajectory = open_file('Punto 2\\data\\trajectory.txt')
    trajectory = np.array(trajectory)
    estimated_positions = []
    initial_guess = [0.0, 0.0, 0.0]
    for measurement in measurements:
        solution = newton(system_of_equations, initial_guess, args=(sensor_positions, measurement))
        estimated_positions.append(solution)
        initial_guess = solution
    estimated_positions = np.array(estimated_positions)
    num_positions = len(estimated_positions)
    times = np.linspace(0, num_positions-1, num_positions)
    cs_x = CubicSpline(times, estimated_positions[:, 0])
    cs_y = CubicSpline(times, estimated_positions[:, 1])
    cs_z = CubicSpline(times, estimated_positions[:, 2])
    interpolated_times = np.linspace(times.min(), times.max(), len(trajectory))
    interpolated_trajectory = np.vstack((cs_x(interpolated_times), cs_y(interpolated_times), cs_z(interpolated_times))).T
    graphs.plot_trajectory_3d(trajectory, interpolated_trajectory, estimated_positions)
    errores = calculate_interpolation_error(trajectory[:, 1:], interpolated_trajectory)
    graphs.plot_interpolation_error(errores, interpolated_times)

if __name__ == "__main__":
    main()
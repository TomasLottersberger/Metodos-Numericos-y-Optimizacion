import numpy as np
from function import fb, chebyshevNodes
from errors import calculate_error, max_error_3d
import matplotlib.pyplot as plt
from button import ButtonManager
from matplotlib.widgets import Button
from scipy.interpolate import RegularGridInterpolator

def show_points_plot():
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    xg, yg = np.meshgrid(x, y, indexing="xy")

    data = fb([xg, yg])
    new_x = np.linspace(-1, 1, 100)
    new_y = np.linspace(-1, 1, 100)
    
    fig = plt.figure()
    temp=''
    button_manager = ButtonManager(xg, yg, data, temp, temp, temp, temp, fig, temp)

    ax_button_points = plt.axes([0.59, 0.01, 0.1, 0.05])
    button_points = Button(ax_button_points, 'Points')
    button_points.on_clicked(button_manager.data_points)

    button_manager.on_points_click()
    plt.show()

def show_function_plot():
    new_x = np.linspace(-1, 1, 100)
    new_y = np.linspace(-1, 1, 100)

    X, Y = np.meshgrid(new_x, new_y, indexing="xy")

    Z_ground_truth = fb([X, Y])

    fig = plt.figure()
    temp = ''
    button_manager = ButtonManager(temp, temp, temp, X, Y, temp, Z_ground_truth, fig,temp)


    ax_button_ground_truth = plt.axes([0.81, 0.01, 0.1, 0.05])
    button_ground_truth = Button(ax_button_ground_truth, 'Ground Truth')
    button_ground_truth.on_clicked(button_manager.on_ground_truth_click)

    button_manager.on_ground_truth_click(None)
    plt.show()
    
    
    
def show_error(use_chebyshev=False, use_median=False, method='cubic', num_points_eval=50):
    x_range = [-1, 1]
    y_range = [-1, 1]
    max_errors = []
    num_points_list = []
    new_x = np.linspace(-1, 1, num_points_eval)
    new_y = np.linspace(-1, 1, num_points_eval) 
    X, Y = np.meshgrid(new_x, new_y, indexing="xy")

    for num_points in range(4, 20, 1):
        if use_chebyshev:
            x = chebyshevNodes(num_points, x_range[0], x_range[1])
            y = chebyshevNodes(num_points, y_range[0], y_range[1])
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)

        xg, yg = np.meshgrid(x, y, indexing="xy")
        data = fb([xg, yg])
        interp = RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=None, method=method)
        Z_interp = interp((X, Y))
        Z_ground_truth = fb([X, Y])        
        max_err = max_error_3d(Z_ground_truth, Z_interp, use_median)
        max_errors.append(max_err)
        print(f"Para {num_points} puntos, el error máximo es: {max_err}")
        num_points_list.append(num_points)

    plt.figure(figsize=(10, 6))
    plt.bar(num_points_list, max_errors)
    plt.xlabel("Número de puntos")
    if use_median:
        plt.ylabel("Error Mediano")

        plt.title("Error Mediano vs. Número de puntos")
    else:
        plt.title("Error Maximo vs. Número de puntos")
        plt.ylabel("Error Maximo")
    plt.yscale("log")
    plt.show()
    
def show_error2(use_chebyshev=False, use_median=False, method='linear', num_points_eval=50):
    x_range = [-1, 1]
    y_range = [-1, 1]
    max_errors = []
    median_errors = []
    num_points_list = []
    new_x = np.linspace(-1, 1, num_points_eval)
    new_y = np.linspace(-1, 1, num_points_eval) 
    X, Y = np.meshgrid(new_x, new_y, indexing="xy")

    for num_points in range(4, 40, 1):
        if use_chebyshev:
            x = chebyshevNodes(num_points, x_range[0], x_range[1])
            y = chebyshevNodes(num_points, y_range[0], y_range[1])
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)

        xg, yg = np.meshgrid(x, y, indexing="xy")
        data = fb([xg, yg])
        interp = RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=None, method=method)
        Z_interp = interp((X, Y))
        Z_ground_truth = fb([X, Y])        
        max_err = max_error_3d(Z_ground_truth, Z_interp, False)
        median_err = max_error_3d(Z_ground_truth, Z_interp, True)
        max_errors.append(max_err)
        median_errors.append(median_err)
        print(f"Para {num_points} puntos, el error máximo es: {max_err}, el error mediano es: {median_err}")
        num_points_list.append(num_points)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.bar(num_points_list, max_errors)
    ax1.set_ylabel("Error Máximo",fontsize=25)

    ax2.bar(num_points_list, median_errors)
    ax2.set_ylabel("Error Mediano",fontsize=25)

    plt.tight_layout()
    plt.show()

def show_both(use_chebyshev=False, method='cubic', num_points_construct=10, num_points_eval=100):
    if use_chebyshev:
            x = chebyshevNodes(num_points_construct, -1, 1)
            y = chebyshevNodes(num_points_construct, -1, 1)
    else:
            x = np.linspace(-1, 1, num_points_construct)
            y = np.linspace(-1, 1, num_points_construct)

    xg, yg = np.meshgrid(x, y, indexing="xy")
    data = fb([xg, yg])
    interp = RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=None, method=method)
    new_x = np.linspace(-1, 1, num_points_eval)
    new_y = np.linspace(-1, 1, num_points_eval)
    X, Y = np.meshgrid(new_x, new_y, indexing="xy")
    Z_interp = interp((X, Y))
    Z_ground_truth = fb([X, Y])

    fig = plt.figure()
    button_manager = ButtonManager(xg, yg, data, X, Y, Z_interp, Z_ground_truth, fig, method)

    ax_button_interpolation = plt.axes([0.7, 0.01, 0.1, 0.05])
    button_interpolation = Button(ax_button_interpolation, 'Interpolacion Lineal')
    button_interpolation.on_clicked(button_manager.on_interpolation_click)

    ax_button_ground_truth = plt.axes([0.81, 0.01, 0.1, 0.05])
    button_ground_truth = Button(ax_button_ground_truth, 'Ground Truth')
    button_ground_truth.on_clicked(button_manager.on_ground_truth_click)

    ax_button_both = plt.axes([0.92, 0.01, 0.1, 0.05])
    button_both = Button(ax_button_both, 'Both')
    button_both.on_clicked(button_manager.on_both_click)

    button_manager.on_both_click(None)
    plt.show()

def show_interpolation(use_chebyshev=False, method='cubic', num_points_construct=10, num_points_eval=100):
    if use_chebyshev:
        x = chebyshevNodes(num_points_construct, -1, 1)
        y = chebyshevNodes(num_points_construct, -1, 1)
    else:
        x = np.linspace(-1, 1, num_points_construct)
        y = np.linspace(-1, 1, num_points_construct)

    xg, yg = np.meshgrid(x, y, indexing="xy")
    data = fb([xg, yg])
    interp = RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=None, method=method)

    new_x = np.linspace(-1, 1, num_points_eval)
    new_y = np.linspace(-1, 1, num_points_eval)
    X, Y = np.meshgrid(new_x, new_y, indexing="xy")
    Z_interp = interp((X, Y))
    Z_ground_truth = fb([X, Y])

    fig = plt.figure()
    button_manager = ButtonManager(xg, yg, data, X, Y, Z_interp, Z_ground_truth, fig, method)

    ax_button_interpolation = plt.axes([0.7, 0.01, 0.1, 0.05])
    button_interpolation = Button(ax_button_interpolation, 'Interpolation',)
    button_interpolation.on_clicked(button_manager.on_interpolation_click)

    ax_button_ground_truth = plt.axes([0.81, 0.01, 0.1, 0.05])
    button_ground_truth = Button(ax_button_ground_truth, 'Ground Truth')
    button_ground_truth.on_clicked(button_manager.on_ground_truth_click)

    button_manager.on_interpolation_click(None)
    plt.show()

def main():
    #show_function_plot()
    #show_points_plot()
    #show_interpolation()
    #show_interpolation(use_chebyshev=True)
    #show_both()
    #show_both(use_chebyshev=True)
    #show_error()
    #show_error(use_chebyshev=True,use_median=False)
    #show_error(use_chebyshev=False, use_median=True)
    #show_error(use_chebyshev=True, use_median=True)
    #show_interpolation(method='linear')
    #show_interpolation(use_chebyshev=True,method='linear')
    #show_both(method='linear')
    #show_both(use_chebyshev=True,method='linear')
    #show_error(method='linear')
    ##show_error(use_chebyshev=True,use_median=False,method='linear')
    #show_error(use_chebyshev=False, use_median=True,method='linear')
    #show_error(use_chebyshev=True, use_median=True,method='linear')
    #heatmap(method='linear')
    #heatmap(use_chebyshev=True,method='linear')
    #show_both(method='linear')
    show_error2(method="cubic")

    
if __name__ == "__main__":

    main()
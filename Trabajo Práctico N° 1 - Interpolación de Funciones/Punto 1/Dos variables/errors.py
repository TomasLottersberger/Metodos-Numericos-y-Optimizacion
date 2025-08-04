import numpy as np

def calculate_error(Z_ground_truth, Z_interp):
    # Calculate the absolute error
    error = np.abs(Z_ground_truth - Z_interp)
    return error

def max_error_3d(Z_ground_truth, Z_interp, use_median):
    # Calculate the error
    error = calculate_error(Z_ground_truth, Z_interp)
    # Encontrar el error máximo
    if use_median:
         max_error = np.median(error)
    else:
        max_error = np.max(error)
    return max_error


import numpy as np

def pseudoinverse_solution(X, y):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = np.diag(1 / Sigma)
    X_pseudo = VT.T @ Sigma_inv @ U.T
    w = X_pseudo @ y
    return w

def gradient_descent(X_train, X_test, y_train, y_test, eta, max_iter=30000, tol=1e-6):
    iterations = 0
    n, d = X_train.shape
    w_gd = np.zeros(d)
    mse_train_history = []
    mse_test_history = []
    w = np.zeros(d)
    trajectory = [w.copy()]
    for iteration in range(max_iter):
        iterations +=1
        gradient = -2 / n * X_train.T @ (y_train - X_train @ w_gd)
        w_gd -= eta * gradient
        mse_train = np.mean((y_train - X_train @ w_gd) ** 2)
        mse_test = np.mean((y_test - X_test @ w_gd) ** 2)
        mse_train_history.append(mse_train)
        mse_test_history.append(mse_test)
        trajectory.append(w.copy())
        if np.linalg.norm(gradient) < tol:
            break
        #print(iterations) opt = 10k
    return w_gd, mse_train_history, mse_test_history, trajectory

def gradient_descent_analysis(X, y, eta, max_iter=5000, tol=1e-6):
    n, d = X.shape
    w = np.zeros(d)
    errors = []
    for t in range(max_iter):
        gradient = -2 / n * X.T @ (y - X @ w)
        w -= eta * gradient
        error = np.mean((y - X @ w)**2)
        errors.append(error)
        if np.linalg.norm(gradient) < tol:
            break
    return errors

def optimum_learning_rate(X_train):
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    sigma_1 = S[0]
    eta_opt = 1 / (sigma_1**2)
    return eta_opt
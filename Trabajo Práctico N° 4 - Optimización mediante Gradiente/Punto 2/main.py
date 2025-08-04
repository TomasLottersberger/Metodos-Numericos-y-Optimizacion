import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functions
import graphs

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])
    w_pseudo = functions.pseudoinverse_solution(X_train, y_train)
    y_pred_pseudo_train = X_train @ w_pseudo
    y_pred_pseudo_test = X_test @ w_pseudo
    mse_pseudo_train = np.mean((y_train - y_pred_pseudo_train) ** 2)
    mse_pseudo_test = np.mean((y_test - y_pred_pseudo_test) ** 2)
    eta = functions.optimum_learning_rate(X_train)
    w_gd, mse_train_history, mse_test_history, trajectory = functions.gradient_descent(X_train, X_test, y_train, y_test, eta)
    graphs.learning_rates_comparation(X_train, y_train, eta)
    #graphs.mse_for_pseudoinverse_graph(mse_pseudo_train, mse_pseudo_test)
    #graphs.mse_for_gd_graph(mse_train_history, mse_test_history)
    #graphs.mse_per_iteration_gd_graph(mse_train_history, mse_test_history)
    graphs.solution_comparation_graph(w_pseudo, w_gd)
    print(f'w_pseudo = {w_pseudo}\nw_gd = {w_gd}')
    graphs.trajectory_gd_graph(trajectory, w_pseudo)
    graphs.difference_graph(w_pseudo, w_gd)

if __name__ == "__main__":
    main()
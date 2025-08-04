import numpy as np
import matplotlib.pyplot as plt

#Funcion de Rosenbrock
def rosenbrock(x, y, a=1, b=100):
  return (a - x)**2 + b*(y - x**2)**2

#Gradiente de la funcion de Rosenbrock
def grad_rosenbrock(x, y, a=1, b=100):
  grad_x = -2*(a - x) - 4*b*x*(y - x**2)
  grad_y = 2*b*(y - x**2)
  return np.array([grad_x, grad_y])

#Hessiana de la funcion de Rosenbrock
def hessian_rosenbrock(x, y, a=1, b=100):
  hess_xx = 2 - 4*b*(y - x**2) + 8*b*x**2
  hess_xy = -4*b*x
  hess_yy = 2*b
  return np.array([[hess_xx, hess_xy], [hess_xy, hess_yy]])

#Gradiente descendente
def gradient_descent(x0, y0, learning_rate, tolerance, max_iter):
  x, y = x0, y0
  trajectory = [(x, y)]
  for i in range(max_iter):
    grad = grad_rosenbrock(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    trajectory.append((x, y))
    if np.linalg.norm(grad) < tolerance:
      break
  return x, y, trajectory, i+1

#Metodo de Newton (Opcional)
def newton_method(x0, y0, tolerance, max_iter):
  x, y = x0, y0
  trajectory = [(x, y)]
  for i in range(max_iter):
    grad = grad_rosenbrock(x, y)
    hess = hessian_rosenbrock(x, y)
    inv_hess = np.linalg.inv(hess)
    x -= (inv_hess @ grad)[0]
    y -= (inv_hess @ grad)[1]
    trajectory.append((x, y))
    if np.linalg.norm(grad) < tolerance:
      break
  return x, y, trajectory, i+1

#graficar curvas de nivel y trayectorias con los distintos learning rates
def plot_contour_with_trajectories(trajectories, labels, title):
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    #add multiple trajectories
    for trajectory, label in zip(trajectories, labels):
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1],'-' ,label=label)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.colorbar()
    plt.show()

#Punto 1
print("Gradiente descendente")
#Valores iniciales
x0, y0 = -1.5, 2.0
er = 1e-3
max_iter = 100000
#Learning rates
learning_rates = [0.001,0.0015,0.0017,0.002]
trajectories = []
labels = []
for lr in learning_rates:
    x ,y , trajectory, it = gradient_descent(x0, y0, lr, er, max_iter)
    trajectories.append(trajectory)
    labels.append(f'Learning rate: {lr}')
    print(f'Learning rate: {lr}, Punto de convergencia: ({x:.2f}, {y:.2f}), Iteraciones: {it}')
plot_contour_with_trajectories(trajectories, labels, 'Gradiente descendente con distintos learning rates')




#grafico de learning rates vs iteraciones
learning_rates = np.linspace(0.001, 0.00208, 100)
trajectories = [gradient_descent(x0, y0, lr, er, max_iter)[2] for lr in learning_rates]
#muestro error y max iteraciones
textstr = '\n'.join((
    r'$\epsilon=%.0e$' % (er, ),))
plt.figure(figsize=(8, 6))
plt.plot(learning_rates, [len(t) for t in trajectories], '-') 
plt.text(0.001, 100, textstr, fontsize=12)
plt.title('Learning rates vs Iteraciones')
plt.xlabel('Learning rate')
plt.ylabel('Iteraciones')
plt.show()

#printear el learning rate con menor cantidad de iteraciones
min_iter = min([len(t) for t in trajectories])
min_lr = learning_rates[[len(t) for t in trajectories].index(min_iter)]
print(f'Learning rate con menor cantidad de iteraciones: {min_lr}, Iteraciones: {min_iter}')

#calcular los errores cuadraticos medios para los learning rates [0.001,0.0015,0.0017,0.002] vs iteraciones
#graficar el error cuadratico medio vs iteraciones
#y error cuadratico medio para el learning rates
#x numero de iteraciones
learning_rates = [0.001,0.0015,0.0017,0.002, min_lr]
trajectories = [gradient_descent(x0, y0, lr, er, max_iter)[2] for lr in learning_rates]
errors = [np.mean([rosenbrock(x, y) for x, y in t]) for t in trajectories]
min_error = np.mean([rosenbrock(x, y) for x, y in gradient_descent(x0, y0, min_lr, er, max_iter)[2]])
#grafico de error cuadratico medio vs iteraciones en escala logaritmica
plt.figure(figsize=(8, 6))
for i, error in enumerate(errors):
    plt.plot(range(len(trajectories[i])), [rosenbrock(x, y) for x, y in trajectories[i]], '-', label=f'Learning rate: {learning_rates[i]}')
plt.yscale('log')
plt.title('Error cuadratico medio vs Iteraciones')
plt.xlabel('Iteraciones')
plt.ylabel('Error cuadratico medio')
plt.legend()
plt.show()


#Punto 2 pero utilizando metodo de Newton
print("Metodo de Newton")
x ,y , trajectory, it = newton_method(x0, y0, er, 100000)
print(f'Punto de convergencia: ({x:.2f}, {y:.2f}), Iteraciones: {it}')
plot_contour_with_trajectories([trajectory], ['Newton Method'], 'Metodo de Newton')

#misma learning rate pero con distintos valores iniciales
x0 = [0,0.5,1.5,2]
y0 = [0,0.5,1.5,2]
trajectories = []
labels = []
for i in range(4):
    x ,y , trajectory, it = gradient_descent(x0[i], y0[i], 0.0015, er, max_iter)
    trajectories.append(trajectory)
    labels.append(f'Punto inicial: ({x0[i]}, {y0[i]})')
    print(f'Punto inicial: ({x0[i]}, {y0[i]}), Punto de convergencia: ({x:.2f}, {y:.2f}), Iteraciones: {it}')
plot_contour_with_trajectories(trajectories, labels, 'Gradiente descendente con distintos puntos iniciales')
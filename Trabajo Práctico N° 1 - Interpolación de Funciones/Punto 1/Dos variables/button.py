import numpy as np
import matplotlib.pyplot as plt
from function import fb

class ButtonManager():

    def __init__(self, xg, yg, data, X, Y, Z_interp, Z_ground_truth, fig, method):
        self.xg = xg
        self.yg = yg
        self.data = data
        self.X = X
        self.Y = Y
        self.Z_interp = Z_interp
        self.Z_ground_truth = Z_ground_truth
        self.fig = fig
        self.method = method

    def on_interpolation_click(self, event):
        plt.clf()

        ax = self.fig.add_subplot(projection='3d')
        # interpolator
        ax.plot_wireframe(self.X, self.Y, self.Z_interp, rstride=3, cstride=3,
                        alpha=0.4, color='r', label=f'{self.method} interpolation')
        
        plt.legend()
        plt.draw()

    def on_ground_truth_click(self, event):
        plt.clf()
        ax = self.fig.add_subplot(projection='3d')
        ax.plot_wireframe(self.X, self.Y, self.Z_ground_truth, rstride=3, cstride=3,
                        alpha=0.8, color='b', label='Ground Truth')
        
        plt.legend()
        plt.draw()
    
    def on_both_click(self, event):
        plt.clf()
        ax = self.fig.add_subplot(projection='3d')

        # interpolator
        ax.plot_wireframe(self.X, self.Y, self.Z_interp, rstride=3, cstride=3,
                        alpha=0.6, color='r', label=f'{self.method} Interpolation')

        # ground truth
        ax.plot_wireframe(self.X, self.Y, self.Z_ground_truth, rstride=3, cstride=3,
                        alpha=0.2, color='b', label='Ground Truth')
        
        plt.legend()
        plt.draw()
    
    def on_points_click(self):
        plt.clf()
        ax = self.fig.add_subplot(projection='3d')

        # data
        ax.scatter(self.xg.ravel(), self.yg.ravel(), self.data.ravel(),
                s=10, c='k', label='data')
        
        plt.legend()
        plt.draw()

    def data_points(self):
        # Define data points of the interpolation
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)

        xg, yg = np.meshgrid(x, y, indexing="ij")

        data = fb([xg, yg])

        return xg, yg, data
    
    
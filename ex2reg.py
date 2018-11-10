from utils import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from timeit import timeit
from mpl_toolkits.mplot3d import Axes3D


class Ex2Reg:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
        self.l = None # Regularization parameter
        self.m = None # Number of training examples
        self.n = None # Number of features
        self.mu = None # Features' mean values
        self.sigma = None # Features' std dev values
        self.theta = None # Fitting parameters vector
        self.num_iters = None # Number of iterations
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

        # Init
        self.data = pd.read_csv("ex2-data/ex2data2.txt", header=None)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1][:, np.newaxis]
        self.m, self.n = self.X.shape
        ones = np.ones((self.m, 1))
        self.X = np.hstack((ones, self.X))

        self.plot_data(self.X, self.y)

    def run(self):
        self.regularized_logistic_regression()

    def plot_data(self, X, y):
        fig, ax = plt.subplots()
        positive = y[:, 0] == 1
        negative = ~positive
        ax.plot(X[positive, 1], X[positive, 2], 'r+', linewidth=2, markersize=7, label="y = 1")
        ax.plot(X[negative, 1], X[negative, 2], 'bo', linewidth=2, markersize=7, label="y = 0")
        ax.legend()
        ax.set_xlabel("Microchip Test 1")
        ax.set_ylabel("Microchip Test 2")
        fig.savefig("ex2-data/classes_plot_reg.png")

    def regularized_logistic_regression(self):
        self.X = self.map_feature(self.X[:,1], self.X[:,2])
        self.n = self.X.shape[1]
        self.theta = np.zeros((self.n, 1))
        self.l = 1

    def map_feature(self, X1, X2, degree=6):
        """Feature mapping function to polynomial features"""
        X1 = X1[:, np.newaxis]
        X2 = X2[:, np.newaxis]
        X = np.ones((self.m, 1))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                X = np.hstack((X, X1**(i - j) * X2**i))
        return X


if __name__ == '__main__':
    ex2reg = Ex2Reg()
    sys.exit(ex2reg.run())
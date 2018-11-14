from utils import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from ex2 import Ex2
from scipy.optimize import minimize
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
        self.data = pd.read_csv("ex2reg-data/ex2data2.txt", header=None)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1][:, np.newaxis]
        self.m, self.n = self.X.shape
        ones = np.ones((self.m, 1))
        self.X = np.hstack((ones, self.X))

        self.plot_data(self.X, self.y)

    def run(self):
        self.regularized_logistic_regression()
        self.regularization_and_accuracies()

    def plot_data(self, X, y):
        fig, ax = plt.subplots()
        positive = y[:, 0] == 1
        negative = ~positive
        ax.plot(X[positive, 1], X[positive, 2], 'r+', linewidth=2, markersize=7, label="y = 1")
        ax.plot(X[negative, 1], X[negative, 2], 'bo', linewidth=2, markersize=7, label="y = 0")
        ax.legend()
        ax.set_xlabel("Microchip Test 1")
        ax.set_ylabel("Microchip Test 2")
        fig.savefig("ex2reg-data/classes_plot_reg.png")

    @print_name
    # @pause_after
    def regularized_logistic_regression(self):
        self.X = self.map_feature(self.X[:,1], self.X[:,2])
        self.n = self.X.shape[1]
        self.theta = np.zeros((self.n, 1))
        self.l = 1
        cost = self.cost_function_reg(self.theta, self.X, self.y, self.l)[0,0]
        grad = self.gradient_reg(self.theta, self.X, self.y, self.l)
        print("Cost at initial theta (zeros): {:.4f}".format(cost))
        print("Expected cost (approx): 0.693")
        print("Gradient at initial theta (zeros) - first five values only:")
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*grad[:5]))
        print("Expected gradients (approx) - first five values only:")
        print("0.0085 0.0188 0.0001 0.0503 0.0115")

        # Compute and display cost and gradient with all-ones theta and lambda = 10
        theta = np.ones((self.n, 1))
        l = 10
        cost = self.cost_function_reg(theta, self.X, self.y, l)[0,0]
        grad = self.gradient_reg(theta, self.X, self.y, l)
        print("\nCost at test theta (all-ones) with lambda = 10: {:.4f}".format(cost))
        print("Expected cost (approx): 3.16")
        print("Gradient at test theta (all-ones) - first five values only:")
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*grad[:5]))
        print("Expected gradients (approx) - first five values only:")
        print("0.3460 0.1614 0.1948 0.2269 0.0922")

    def map_feature(self, X1, X2, degree=6):
        """Feature mapping function to polynomial features"""
        if X1.shape:
            m = X1.shape[0]
            X1 = X1[:, np.newaxis]
            X2 = X2[:, np.newaxis]
            X = np.ones((m, 1))
        else:
            X = np.ones(1)
        for i in range(1, degree + 1):
            for j in range(i + 1):
                X = np.hstack((X, X1**(i - j) * X2**j))
        return X

    @staticmethod
    def gradient_reg(theta, X, y, l):
        m = len(y)
        h_x = sigmoid(X @ theta)
        grad = 1 / m * X.T @ (h_x - y)
        grad[1:] += l/m * theta[1:]
        return grad.flatten()

    @staticmethod
    def cost_function_reg(theta, X, y, l):
        m = len(y)
        h_x = sigmoid(X @ theta)
        J = 1/m * np.sum(-y.T @ np.log(h_x) - (1 - y.T) @ np.log(1 - h_x)) + l/(2 * m) * (theta[1:].T @ theta[1:])
        return J

    @print_name
    # @pause_after
    def regularization_and_accuracies(self):
        self.theta = np.zeros((self.n, 1))
        self.l = 1
        solver = 'BFGS'
        res = minimize(fun=self.cost_function_reg,
                       x0=self.theta,
                       jac=self.gradient_reg,
                       args=(self.X, self.y.flatten(), self.l),
                       method=solver)
        self.theta = res.x

        self.plot_decision_boundry(self.theta, self.X, self.y)

        predicted = Ex2.predict(self.theta[:,np.newaxis], self.X)
        accuracy = np.mean(predicted == self.y) * 100
        print("Train Accuracy (with lambda = {}): {:.2f}%".format(self.l, accuracy))
        print("Expected accuracy (with lambda = 1): 83.1% (approx)")

    def plot_decision_boundry(self, theta, X, y):
        fig, ax = plt.subplots()
        positive = y[:, 0] == 1
        negative = ~positive
        ax.plot(X[positive, 1], X[positive, 2], 'r+', linewidth=2, markersize=7, label="y = 1")
        ax.plot(X[negative, 1], X[negative, 2], 'bo', linewidth=2, markersize=7, label="y = 0")
        ax.legend()
        ax.set_xlabel("Microchip Test 1")
        ax.set_ylabel("Microchip Test 2")

        # plot decision boundry
        u = np.linspace(-1, 1.5, 50)
        v = u.copy()
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = self.map_feature(u[i], v[j]) @ theta
        # Contour plot
        ax.contour(u, v, z, 0)
        ax.set_title("lambda = " + str(self.l))
        fig.savefig("ex2reg-data/decision_boundry_reg.png")


if __name__ == '__main__':
    ex2reg = Ex2Reg()
    sys.exit(ex2reg.run())
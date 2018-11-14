from utils import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from timeit import timeit
from mpl_toolkits.mplot3d import Axes3D


class Ex2:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
        self.m = None # Number of training examples
        self.n = None # Number of features
        self.mu = None # Features' mean values
        self.sigma = None # Features' std dev values
        self.theta = None # Fitting parameters vector
        self.num_iters = None # Number of iterations
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

        # Init
        self.data = pd.read_csv("ex2-data/ex2data1.txt", header=None)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1][:, np.newaxis]
        self.m, self.n = self.X.shape
        ones = np.ones((self.m, 1))
        self.X = np.hstack((ones, self.X))

    def run(self):
        self.plotting()
        self.compute_cost_and_gradient()
        self.optimizing()
        self.predict_and_accuracy()

    @print_name
    # @pause_after
    def plotting(self):
        print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
        self.plot_data(self.X, self.y)

    def plot_data(self, X, y):
        fig, ax = plt.subplots()
        admitted = y[:, 0] == 1
        not_admitted = ~admitted
        ax.plot(X[admitted, 1], X[admitted, 2], 'r+', linewidth=2, markersize=7, label="Admitted")
        ax.plot(X[not_admitted, 1], X[not_admitted, 2], 'bo', linewidth=2, markersize=7, label="Not admitted")
        ax.legend()
        ax.set_xlabel("Exam 1 score")
        ax.set_ylabel("Exam 2 score")
        fig.savefig("ex2-data/classes_plot.png")

    @print_name
    # @pause_after
    def compute_cost_and_gradient(self):
        # Init theta
        self.theta = np.zeros((self.n + 1, 1))
        cost, grad = (self.cost_function(self.theta, self.X, self.y), self.gradient(self.theta, self.X, self.y))
        print("Cost at initial theta (zeros): {:.4f}".format(cost))
        print("Expected cost (approx): 0.693".format(cost))
        print("Gradient at initial theta:")
        print("[{:.4f} {:.4f} {:.4f}]".format(*grad))
        print("Expected gradients:")
        print("[-0.1000 -12.0092 -11.2628]")

        # Compute and display cost and gradient with non-zero theta
        theta = np.reshape([-24, 0.2, 0.2], (3, 1))
        cost, grad = (self.cost_function(theta, self.X, self.y), self.gradient(theta, self.X, self.y))
        print("\nCost at thest theta=[-24 0.2 0.2]: {:.4f}".format(cost))
        print("Expected cost (approx): 0.218")
        print("Gradient at test theta:")
        print("[{:.4f} {:.4f} {:.4f}]".format(*grad))
        print("Expected gradients:")
        print("[0.043 2.566 2.647]")

    @staticmethod
    def gradient(theta, X, y):
        m = len(y)
        h_x = sigmoid(X @ theta)
        grad = 1/m * X.T @ (h_x - y)
        return grad.flatten()

    @staticmethod
    def cost_function(theta, X, y):
        m = len(y)
        h_x = sigmoid(X @ theta)
        J = 1/m * np.sum(-y.T @ np.log(h_x) - (1 - y.T) @ np.log(1 - h_x))
        return J

    @print_name
    # @pause_after
    def optimizing(self):
        # Init theta
        self.theta = np.zeros((self.n + 1, 1))
        solver = 'BFGS'
        res = minimize(fun=self.cost_function,
                       x0=self.theta,
                       jac=self.gradient,
                       args=(self.X, self.y.flatten()),
                       method=solver)
        self.theta = res.x
        print("Cost at theta found by unconstrained minimization algorithm:")
        print("{}: {:.4f}".format(solver, res.fun))
        print("Expected cost (approx): 0.203")
        print("Calculated theta:")
        print("{}: [{:.4f} {:.4f} {:.4f}]".format(solver, *res.x))
        print("Expected theta (approx):")
        print("[-25.161 0.206 0.201]")

        self.plot_decision_boundry(self.theta, self.X, self.y)

    def plot_decision_boundry(self, theta, X, y):
        # plot data
        fig, ax = plt.subplots()
        admitted = y[:, 0] == 1
        not_admitted = ~admitted
        ax.plot(X[admitted, 1], X[admitted, 2], 'r+', linewidth=2, markersize=7, label="Admitted")
        ax.plot(X[not_admitted, 1], X[not_admitted, 2], 'bo', linewidth=2, markersize=7, label="Not admitted")
        ax.legend()
        ax.set_xlabel("Exam 1 score")
        ax.set_ylabel("Exam 2 score")

        # plot decision boundry
        bx = np.reshape([np.min(X[:,1]), np.max(X[:,1])], (2, 1))
        by = (theta[0] + theta[1] * bx) / -theta[2]
        ax.plot(bx, by, linewidth=2, label="Decision boundry")
        ax.legend()

        fig.savefig("ex2-data/decision_boundry.png")

    @print_name
    # @pause_after
    def predict_and_accuracy(self):
        prob = sigmoid(np.array([1, 45, 85]) @ self.theta)
        print("For a student with scores 45 and 85, we predict and admission probability of {:.4f}".format(prob))
        print("Expected value: 0.775 +/- 0.002")
        predicted = self.predict(self.theta[:, np.newaxis], self.X)
        accuracy = np.mean(predicted == self.y)
        print("Train Accuracy: {:.4f}".format(accuracy))
        print("Expected accuracy (approx): 0.89")

    @staticmethod
    def predict(theta, X):
        return np.round(sigmoid(X @ theta))


if __name__ == '__main__':
    ex2 = Ex2()
    sys.exit(ex2.run())
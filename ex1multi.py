from utils import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class Ex1Multi:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
        self.m = None # Number of training examples
        self.mu = None # Features' mean values
        self.sigma = None # Features' std dev values
        self.theta = None # Fitting parameters vector
        self.num_iters = None # Number of iterations
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

    def run(self):
        self.load_and_normalize()
        self.gradient_descent()
        self.normal_equations()

    @print_name
    # @pause_after
    def load_and_normalize(self):
        print("Load data ...")
        self.data = pd.read_csv("ex1multi-data/ex1data2.txt", header=None)
        self.X = self.data.iloc[:, :-1] # All but last column to X
        self.y = self.data.iloc[:, -1] # Last column to y
        self.m = len(self.data)
        print("First 10 examples from the dataset:")
        [print("x = [{} {}], y = {}".format(*row)) for i, row in self.data.head(10).iterrows()]
        self.X, self.mu, self.sigma = self.feature_normalization(self.X)
        # Add intercept term
        ones = np.ones([self.m, 1])
        self.X = np.hstack([ones, self.X])

    def feature_normalization(self, X):
        mu = np.mean(X)
        sigma = np.std(X)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma

    @print_name
    # @pause_after
    def gradient_descent(self):
        self.alpha = 0.03
        self.num_iters = 400
        self.theta = np.zeros([3, 1])
        self.y = self.y[:, np.newaxis]
        self.theta, self.J_history = self.gradient_descent_multi(self.X, self.y, self.theta, self.alpha, self.num_iters)
        self.plot_convergence(self.J_history)
        print("Theta computed from gradient descent:")
        print("[{:.2f} {:.2f} {:.2f}]".format(*np.squeeze(self.theta)))
        # Estimate the price of a 1650 sq-ft, 3 br house
        A = [1, 1650, 3]
        for i in range(1, len(A)):
            A[i] = (A[i] - self.mu[i - 1]) / self.sigma[i - 1]
        price = np.dot(self.theta.T, A)
        print("Predicted price of a 1650 sq-ft, 3 br house using gradient descent:")
        print("{:.2f}".format(price[0]))

    def gradient_descent_multi(self, X, y, theta, alpha, num_iters):
        m = len(self.y)
        n = len(theta)
        J_history = np.zeros([num_iters, 1])
        for i in range(num_iters):
            err = np.dot(X, theta) - y
            theta_new = theta - alpha/m * np.dot(X.T, err)
            theta = theta_new
            J_history[i] = self.compute_cost_multi(X, y, theta)
        return theta, J_history

    def compute_cost_multi(self, X, y, theta):
        m = len(y)
        return np.sum((np.dot(X, theta) - y)**2) / (2*m)

    def plot_convergence(self, J_history):
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(J_history) + 1), J_history, '-b', linewidth=2)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Cost J")
        fig.savefig("ex1multi-data/convergence.png")

    @print_name
    def normal_equations(self):
        self.X = self.data.iloc[:, :-1]  # Reverse normalization
        # Add intercept term
        ones = np.ones([self.m, 1])
        self.X = np.hstack([ones, self.X])
        self.theta = self.normal_eqn(self.X, self.y)
        print("Theta computed from the normal equations:")
        print("[{:.2f} {:.2f} {:.2f}]".format(*np.squeeze(self.theta)))
        price = self.theta.T @ [1, 1650, 3]
        print("Predicted price of a 1650 sq-ft, 3 br house using normal equations:")
        print("{:.2f}".format(price[0]))

    def normal_eqn(self, X, y):
        theta = np.linalg.inv(X.T @ X) @ X.T @ y # @ instead of np.dot()
        return theta


if __name__ == '__main__':
    ex1multi = Ex1Multi()
    sys.exit(ex1multi.run())
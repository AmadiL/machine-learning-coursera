from utils import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Ex1:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
        self.m = None # Number of training examples
        self.theta = None # Fitting parameters vector
        self.iterations = None # Number of iterations
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

    def run(self):
        self.warm_up_exercise()
        self.plotting()
        self.cost_and_gradient_descent()
        self.visualizing_cost_function()

    @print_name
    # @pause_after
    def warm_up_exercise(self):
        """Print the 5x5 identity matrix"""
        print("5x5 Identity Matrix:")
        print(np.eye(5))

    @print_name
    # @pause_after
    def plotting(self):
        print("Plotting Data ...")
        self.data = pd.read_csv("ex1-data/ex1data1.txt", header=None)
        self.data.columns = ["Population of City in 10,000s", "Profit in $10,000s"]
        self.X = self.data.iloc[:, 0]
        self.y = self.data.iloc[:, 1]
        self.m = len(self.data)
        self.plot_data(self.X, self.y)

    def plot_data(self, x, y):
        """Plot the training data into a figure using "figure" and "plot".
        Set the axes labels. Assume the population and revenue data have been passed in
        as the x and y arguments of this function."""
        fig, ax = plt.subplots()
        ax.plot(x, y, "rx", markersize=10)
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        fig.savefig("ex1-data/plot_data.png")

    @print_name
    # @pause_after
    def cost_and_gradient_descent(self):
        # Add a column of ones to X
        self.X = self.X[:, np.newaxis]
        self.y = self.y[:, np.newaxis]
        ones = np.ones([self.m, 1])
        self.X = np.hstack([ones, self.X])

        # Initialize fitting parameters
        self.theta = np.zeros([2, 1])

        # Gradient descent parameters
        self.iterations = 1500
        self.alpha = 0.01

        # Compute initial cost
        print("Testing the cost function ...")
        J = self.compute_cost(self.X, self.y, self.theta)
        print("With theta = [0, 0]")
        print("Cost computed = " + str(J))
        print("Expected cost value ~32.07")
        theta = np.reshape([-1, 2], (2, 1))
        J = self.compute_cost(self.X, self.y, theta)
        print("With theta = [-1, 2]")
        print("Cost computed = " + str(J))
        print("Expected cost value ~54.24")

        # Run gradient descent
        self.theta, self.J_history = self.gradient_descent(self.X, self.y, self.theta, self.alpha, self.iterations)
        print("Theta found by gradient descent: {} {}".format(*self.theta))
        print("Expected theta values (approx): [-3.6303] [1.1664]")

        # Plot the linear fit
        fig, ax = plt.subplots()
        ax.plot(self.X[:,1], self.y, 'o', label="Training data")
        ax.plot(self.X[:,1], np.dot(self.X, self.theta), 'r', label="Linear regression")
        ax.legend()
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        fig.savefig("ex1-data/linear_fit.png")

        # Predict values for population sizes of 35k and 70k
        predict = np.dot([1, 3.5], self.theta) * 10000
        print("For population = 35k, we predict a profit of " + str(predict))
        predict = np.dot([1, 7], self.theta) * 10000
        print("For population = 70k, we predict a profit of " + str(predict))

    def compute_cost(self, X, y, theta):
        m = len(y)
        return np.sum((np.dot(X, theta) - y)**2) / (2*m)

    def gradient_descent(self, X, y, theta, alpha, num_iters):
        m = len(y)
        n = len(theta)
        J_history = np.zeros([num_iters, 1])
        for i in range(num_iters):
            err = np.dot(X, theta) - y
            theta_new = theta - alpha/m * np.dot(X.T, err)
            theta = theta_new
            J_history[i] = self.compute_cost(X, y, theta)
        return theta, J_history

    def visualizing_cost_function(self):
        # Grid over which we will calculate cost function J(theta_0, theta_1)
        theta0_vals = np.linspace(-10, 10, 100)
        theta1_vals = np.linspace(-1, 4, 100)

        # Initialize J_vals to matrix of 0's
        J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])
        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                theta = [theta0_vals[i], theta1_vals[j]]
                theta = np.reshape(theta, (2,1))
                J_vals[i, j] = self.compute_cost(self.X, self.y, theta)

        # Surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(theta0_vals, theta1_vals, J_vals)
        ax.plot(self.theta[0], self.theta[1], 'rx')
        fig.savefig("ex1-data/cost_function.png")
        # Contour plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
        ax.plot(self.theta[0], self.theta[1], 'rx')
        fig.savefig("ex1-data/cost_function_contour.png")


if __name__ == '__main__':
    ex1 = Ex1()
    sys.exit(ex1.run())
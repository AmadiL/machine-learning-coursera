from utils import *
import sys
import scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.misc import imsave
from ex2reg import Ex2Reg
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


class Ex5:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.Xval = None # X cross-validation matrix
        self.Xtest = None # X test matrix
        self.y = None # y vector
        self.yval = None # y cross-validation vector
        self.ytest = None # y test vector
        self.hidden_layer_size = None # Number of nodes in hidden layer
        self.input_layer_size = None # Number of nodes in input layer
        self.m = None # Number of training examples
        self.n = None # Number of features
        self.mu = None # Features' mean values
        self.reg = None  # Regularization parameter
        self.sigma = None # Features' std dev values
        self.theta1 = None # Fitting parameters for layer 1
        self.theta2 = None # Fitting parameters for layer 2
        self.initial_nn_params = None # Initial fitting parameters
        self.nn_params = None # Fitting parameters
        self.num_iters = None # Number of iterations
        self.num_labels = None # Number of labels / classes
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

        # Init

    def run(self):
        self.loading_and_visualizing_data()
        self.regularized_linear_regression_cost()
        self.regularized_linear_regression_gradient()
        self.train_linear_regression()
        self.learning_curve_for_linear_regression()
        self.feature_mapping_and_polynomial_regression()

    @print_name
    # @pause_after
    def loading_and_visualizing_data(self):
        self.data = sio.loadmat("ex5-data/ex5data1.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.Xval = self.data['Xval']
        self.yval = self.data['yval']
        self.Xtest = self.data['Xtest']
        self.ytest = self.data['ytest']
        self.m, self.n = self.X.shape

        fig, ax = plt.subplots()
        ax.plot(self.X, self.y, 'rx', markersize=10, linewidth=1.5)
        ax.set_xlabel("Change in water level (x)")
        ax.set_ylabel("Water flowing out of the dam (y)")
        fig.savefig("ex5-data/training_data_plot.png")

    @print_name
    # @pause_after
    def regularized_linear_regression_cost(self):
        X = np.hstack((np.ones((self.m, 1)), self.X))
        theta = np.ones((2,1))
        reg = 1
        cost, grad = self.linear_reg_cost_function(theta, X, self.y, reg)
        print("Cost at theta = [1, 1]: {:.4f}".format(cost))
        print("(this value should be about 303.993192)")

    def linear_reg_cost_function(self, theta, X, y, reg):
        m, n = X.shape
        h_x = X @ theta.reshape((-1, 1))
        err = h_x - y.reshape((-1, 1))

        cost = (err.T @ err + reg * (theta[1:].T @ theta[1:])) / (2 * m)

        grad = X.T @ err / m
        grad[1:] += theta[1:] * reg / m

        return cost[0,0], grad.flatten()

    @print_name
    # @pause_after
    def regularized_linear_regression_gradient(self):
        X = np.hstack((np.ones((self.m, 1)), self.X))
        theta = np.ones((2, 1))
        reg = 1
        cost, grad = self.linear_reg_cost_function(theta, X, self.y, reg)
        print("Gradient at theta = [1, 1]: [{:.4f}; {:.4f}]".format(*grad))
        print("(this value should be about [-15.303016; 598.250744])")

    @print_name
    # @pause_after
    def train_linear_regression(self):
        X = np.hstack((np.ones((self.m, 1)), self.X))
        reg = 0
        theta = self.train_linear_reg(X, self.y, reg)

        fig, ax = plt.subplots()
        ax.plot(self.X, self.y, 'rx', markersize=10, linewidth=1.5)
        ax.plot(self.X, X @ theta, '-', linewidth=2)
        ax.set_xlabel("Change in water level (x)")
        ax.set_ylabel("Water flowing out of the dam (y)")
        fig.savefig("ex5-data/linear_regression_plot.png")

    def train_linear_reg(self, X, y, reg):
        m, n = X.shape
        initial_theta = np.zeros(n)
        solver = 'L-BFGS-B'
        options = {'maxiter': 200, 'disp': False}
        res = minimize(fun=self.linear_reg_cost_function,
                       x0=initial_theta,
                       jac=True,
                       args=(X, y.flatten(), reg),
                       method=solver,
                       options=options)
        return res.x

    @print_name
    # @pause_after
    def learning_curve_for_linear_regression(self):
        m = self.X.shape[0]
        mval = self.Xval.shape[0]
        X = np.hstack((np.ones((m, 1)), self.X))
        Xval = np.hstack((np.ones((mval, 1)), self.Xval))
        reg = 0
        error_train, error_val = self.learning_curve(X, self.y, Xval, self.yval, reg)

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, m + 1), error_train, label='Train')
        ax.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 150)
        ax.legend()
        ax.set_xlabel('Number of training examples')
        ax.set_ylabel('Error')
        fig.savefig("ex5-data/learning_curve.png")

        print("# Training Examples | Train Error | Cross Validation Error")
        for i in range(m):
            print("{:10} {:20.4f} {:20.4f}".format(i + 1, error_train[i], error_val[i]))

    def learning_curve(self, X, y, Xval, yval, reg):
        m = X.shape[0]
        error_train = np.zeros(m)
        error_val = np.zeros(m)
        for i in range(m):
            theta = self.train_linear_reg(X[:i + 1], y[:i + 1], reg)
            error_train[i] = self.linear_reg_cost_function(theta, X[:i + 1], y[:i + 1], 0)[0]
            error_val[i] = self.linear_reg_cost_function(theta, Xval, yval, 0)[0]
        return error_train, error_val

    @print_name
    # @pause_after
    def feature_mapping_and_polynomial_regression(self):
        p = 8
        X_poly = self.poly_features(self.X, p)
        X_poly, mu, sigma = self.feature_normalization(X_poly)
        X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
        print("Normalized Training Example 1:")
        [print("{:7.4f}".format(i)) for i in X_poly[0, :]]

    def poly_features(self, X, p):
        X_poly = X
        for i in range(1, p):
            X_poly = np.hstack((X_poly, X**(i + 1)))
        return X_poly

    def feature_normalization(self, X):
        mu = np.broadcast_to(np.mean(X, 0), X.shape)
        print(mu[0])
        sigma = np.broadcast_to(np.std(X, 0, ddof=1), X.shape)
        print(sigma[0])
        X_norm = (X - mu) / sigma
        return X_norm, mu[0], sigma[0]


if __name__ == '__main__':
    ex5 = Ex5()
    sys.exit(ex5.run())
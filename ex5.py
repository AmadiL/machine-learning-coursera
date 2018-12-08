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
        self.X_poly = None # X matrix with polynomial features
        self.X_poly_test = None # X test matrix with polynomial features
        self.X_poly_val = None # X cross-validation matrix with polynomial features
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
        self.learning_curve_for_polynomial_regression()
        self.validation_for_selecting_lambda()

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
        grad[1:] += (theta[1:] * reg / m).reshape((-1, 1))

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
        options = {'maxiter': 1000, 'disp': False}
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
        # Map X onto polynomial features and normalize
        self.X_poly = self.poly_features(self.X, p)
        self.X_poly, self.mu, self.sigma = self.feature_normalization(self.X_poly)
        self.X_poly = np.hstack((np.ones((self.X_poly.shape[0], 1)), self.X_poly))
        print("Normalized Training Example 1:")
        [print("{:7.4f}".format(i)) for i in self.X_poly[0, :]]

        # Map X_poly_test and normalize (using mu and sigma)
        self.X_poly_test = self.poly_features(self.Xtest, p)
        self.X_poly_test = self.feature_normalization(self.X_poly_test, self.mu, self.sigma)[0]
        self.X_poly_test = np.hstack((np.ones((self.X_poly_test.shape[0], 1)), self.X_poly_test))

        # Map X_poly_val and normalize (using mu and sigma)
        self.X_poly_val = self.poly_features(self.Xval, p)
        self.X_poly_val = self.feature_normalization(self.X_poly_val, self.mu, self.sigma)[0]
        self.X_poly_val = np.hstack((np.ones((self.X_poly_val.shape[0], 1)), self.X_poly_val))

    def poly_features(self, X, p):
        X_poly = X
        for i in range(1, p):
            X_poly = np.hstack((X_poly, X**(i + 1)))
        return X_poly

    def feature_normalization(self, X, mu=None, sigma=None):
        if mu is None:
            mu = np.mean(X, 0)
        mu = np.broadcast_to(mu, X.shape)
        if sigma is None:
            sigma = np.std(X, 0, ddof=1)
        sigma = np.broadcast_to(sigma, X.shape)
        X_norm = (X - mu) / sigma
        return X_norm, mu[0], sigma[0]

    @print_name
    # @pause_after
    def learning_curve_for_polynomial_regression(self):
        p = 8
        reg = 3
        theta = self.train_linear_reg(self.X_poly, self.y, reg)
        x = np.arange(min(self.X) - 15, max(self.X) + 25, 0.5)[:, np.newaxis]
        X = self.poly_features(x, p)
        X = self.feature_normalization(X, self.mu, self.sigma)[0]
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Plot training and fit data
        fig, ax = plt.subplots()
        ax.plot(self.X, self.y, 'rx', markersize=10, linewidth=1.5)
        ax.plot(x, X @ theta, '-', linewidth=2)
        ax.set_title("Polynomial Regression Fit (lambda = {})".format(reg))
        ax.set_xlabel("Change in water level (x)")
        ax.set_ylabel("Water flowing out of the dam (y)")
        fig.savefig("ex5-data/polynomial_regression_plot.png")

        m = self.X_poly.shape[0]
        error_train, error_val = self.learning_curve(self.X_poly, self.y, self.X_poly_val, self.yval, reg)
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, m + 1), error_train, label='Train')
        ax.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title('Polynomial Regression Learning Curve (lambda = {})'.format(reg))
        ax.set_xlabel('Number of training examples')
        ax.set_ylabel('Error')
        fig.savefig("ex5-data/poly_learning_curve.png")

        print("# Training Examples | Train Error | Cross Validation Error")
        for i in range(m):
            print("{:10} {:20.4f} {:20.4f}".format(i + 1, error_train[i], error_val[i]))

    @print_name
    # @pause_after
    def validation_for_selecting_lambda(self):
        reg_vec, error_train, error_val = self.validation_curve(self.X_poly, self.y, self.X_poly_val, self.yval)

        fig, ax = plt.subplots()
        ax.plot(reg_vec, error_train, label='Train')
        ax.plot(reg_vec, error_val, label='Cross Validation')
        ax.legend()
        ax.set_xlabel('lambda')
        ax.set_ylabel('Error')
        fig.savefig("ex5-data/lambda_validation_error.png")

        print("# Training Examples | Train Error | Cross Validation Error")
        for i in range(len(reg_vec)):
            print("{:10.3f} {:20.4f} {:20.4f}".format(reg_vec[i], error_train[i], error_val[i]))

    def validation_curve(self, X, y, Xval, yval):
        reg_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
        error_train = np.zeros(len(reg_vec))
        error_val = np.zeros(len(reg_vec))

        for i in range(len(reg_vec)):
            reg = reg_vec[i]
            theta = self.train_linear_reg(X, y, reg)
            error_train[i] = self.linear_reg_cost_function(theta, X, y, 0)[0]
            error_val[i] = self.linear_reg_cost_function(theta, Xval, yval, 0)[0]
        return reg_vec, error_train, error_val


if __name__ == '__main__':
    ex5 = Ex5()
    sys.exit(ex5.run())
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


class Ex4:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
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
        self.loading_parameters()
        self.compute_cost_feedforward()
        self.evaluating_sigmoid_gradient()
        self.initializing_parameters()
        self.implement_backpropagation()
        self.implement_regularization()
        self.training_nn()
        self.visualize_weights()
        self.implement_predict()

    @print_name
    # @pause_after
    def loading_and_visualizing_data(self):
        self.data = sio.loadmat("ex4-data/ex4data1.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.m, self.n = self.X.shape
        self.num_labels = 10

        # 10x10 images sample matrix
        rand_indices = np.random.permutation(self.m)
        a = 10  # number of images on one row/column
        p = 20  # number of pixels in image row/column
        sample_matrix = np.vstack(np.hsplit(self.X[rand_indices[:a * a], :].reshape(-1, p).T, a))
        imsave("ex4-data/10x10_img_sample.png", sample_matrix)

    @print_name
    # @pause_after
    def loading_parameters(self):
        weights = sio.loadmat("ex4-data/ex4weights.mat")
        self.theta1 = weights["Theta1"]
        self.theta2 = weights["Theta2"]
        self.nn_params = np.concatenate((self.theta1.flatten(), self.theta2.flatten()))

    def compute_cost_feedforward(self):
        self.input_layer_size = self.X.shape[1]
        self.hidden_layer_size = self.theta1.shape[0]
        self.reg = 0
        cost, grad = self.nn_cost_function(self.nn_params, self.input_layer_size, self.hidden_layer_size,
                                     self.num_labels, self.X, self.y, self.reg)
        print("Cost at parameters (loaded from ex4weights) without regularization: {:.6f}".format(cost))
        print("(This value should be about 0.287629)")
        self.reg = 1
        cost, grad = self.nn_cost_function(self.nn_params, self.input_layer_size, self.hidden_layer_size,
                                     self.num_labels, self.X, self.y, self.reg)
        print("\nCost at parameters (loaded from ex4weights) with regularization (lambda = 1): {:.6f}".format(cost))
        print("(This value should be about 0.383770)")

    @staticmethod
    def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg, order='C'):
        theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1), order=order)
        theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1), order=order)

        m = X.shape[0]
        y = pd.get_dummies(y.flatten()).values

        # Feedforward
        a1 = np.hstack((np.ones((m,1)), X))
        z2 = a1 @ theta1.T
        a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
        z3 = a2 @ theta2.T
        a3 = sigmoid(z3)

        # Cost function
        J = -np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3)) / m \
            + reg/(2 * m) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

        # Backpropagation
        delta_3 = a3 - y
        delta_2 = delta_3 @ theta2 * np.hstack((np.ones((z2.shape[0], 1)), sigmoid_gradient(z2)))

        D2 = delta_3.T @ a2
        D1 = delta_2[:, 1:].T @ a1

        theta1_grad = D1/m
        theta2_grad = D2/m

        # Regularization
        theta1_grad[:, 1:] += reg / m * theta1[:, 1:]
        theta2_grad[:, 1:] += reg / m * theta2[:, 1:]

        # Unroll gradients
        grad = np.hstack((theta1_grad.flatten(order=order), theta2_grad.flatten(order=order)))

        return J, grad

    @staticmethod
    def sigmoid_gradient(z):
        return sigmoid(z) * (1 - sigmoid(z))

    @print_name
    # @pause_after
    def evaluating_sigmoid_gradient(self):
        gradient = self.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
        print("Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:")
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*gradient))

    @print_name
    # @pause_after
    def initializing_parameters(self):
        initial_theta1 = self.rand_initialize_weights(self.input_layer_size, self.hidden_layer_size)
        initial_theta2 = self.rand_initialize_weights(self.hidden_layer_size, self.num_labels)
        self.initial_nn_params = np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()))

    @staticmethod
    def rand_initialize_weights(L_in, L_out):
        epsilon_init = 0.12
        W = np.random.uniform(-epsilon_init, epsilon_init, (L_out, L_in + 1))
        return W

    @print_name
    # @pause_after
    def implement_backpropagation(self):
        self.check_nn_gradient()

    def check_nn_gradient(self, reg=0):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        # Generate test data
        theta1 = self.debug_initialize_weights(hidden_layer_size, input_layer_size)
        theta2 = self.debug_initialize_weights(num_labels, hidden_layer_size)

        # Initialize X and y
        X = self.debug_initialize_weights(m, input_layer_size - 1)
        y = 1 + np.mod(np.arange(m) + 1, num_labels)

        # Unroll parameters
        nn_params = np.concatenate((theta1.flatten(1), theta2.flatten(1)))

        # Cost function
        cost_func = lambda p: self.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, reg, order='F')
        cost, grad = cost_func(nn_params)
        numgrad = self.compute_numerical_gradient(cost_func, nn_params)

        # Print two gradient computations
        print(np.column_stack((numgrad, grad)))
        print("The above two columns should be very similar.")
        print("(Left - Numerical Gradient, Right - Analytical Gradient)")

        # Evaluate the norm of the difference between two solutions
        diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
        print("Relative difference should be small (les than 1e-9): {:.4e}".format(diff))


    def debug_initialize_weights(self, fan_out, fan_in):
        W = np.sin(np.arange(fan_out * (fan_in + 1)) + 1).reshape((fan_out, fan_in + 1), order='F') / 10
        return W

    def compute_numerical_gradient(self, cost_func, theta):
        e = 1e-4
        numgrad = np.zeros(theta.size)
        perturb = np.zeros(theta.size)
        for i in range(theta.size):
            perturb[i] = e
            numgrad[i] = (cost_func(theta + perturb)[0] - cost_func(theta - perturb)[0]) / (2 * e)
            perturb[i] = 0
        return numgrad

    @print_name
    # @pause_after
    def implement_regularization(self):
        reg = 3
        debug_cost, debug_grad = self.nn_cost_function(nn_params=self.nn_params,
                                                       input_layer_size=self.input_layer_size,
                                                       hidden_layer_size=self.hidden_layer_size,
                                                       num_labels=self.num_labels,
                                                       X=self.X,
                                                       y=self.y,
                                                       reg=reg)
        print("Cost at (fixed) debugging parameters (w/ lambda = {}): {:.6f}".format(reg, debug_cost))
        print("(for lambda = 3, this value should be about 0.576051)")

    @print_name
    # @pause_after
    def training_nn(self):
        reg = 1
        solver = 'L-BFGS-B'
        options = {'maxiter': 50, 'disp': False}
        res = minimize(fun=self.nn_cost_function,
                       x0=self.initial_nn_params,
                       jac=True,
                       args=(self.input_layer_size, self.hidden_layer_size, self.num_labels, self.X, self.y, reg),
                       method=solver,
                       options=options)
        self.nn_params = res.x
        print("Iteration {} | Cost: {:.4e}".format(options["maxiter"], res.fun))

        # Obtain theta1 and theta2 from nn_params
        self.theta1 = self.nn_params[0:(self.hidden_layer_size * (self.input_layer_size + 1))].reshape(self.hidden_layer_size, (self.input_layer_size + 1))
        self.theta2 = self.nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):].reshape(self.num_labels, (self.hidden_layer_size + 1))

    @print_name
    # @pause_after
    def visualize_weights(self):
        self.display_data(self.theta1[:, 1:])

    @print_name
    # @pause_after
    def implement_predict(self):
        predicted = self.predict(self.theta1, self.theta2, self.X)
        print("Training set accuracy: {:.2f}%".format(np.mean(predicted == self.y.flatten()) * 100))

    def predict(self, theta1, theta2, X):
        m = X.shape[0]
        num_labels = theta2.shape[1]

        h1 = sigmoid(np.hstack((np.ones((m, 1)), X)) @ theta1.T)
        h2 = sigmoid(np.hstack((np.ones((m, 1)), h1)) @ theta2.T)
        return np.argmax(h2, axis=1) + 1

    def display_data(self, X):
        a = 5  # number of images on one row/column
        p = 20  # number of pixels in image row/column
        sample_matrix = np.vstack(np.hsplit(X.reshape(-1, p).T, a))
        imsave("ex4-data/weights.png", sample_matrix)


if __name__ == '__main__':
    ex4 = Ex4()
    sys.exit(ex4.run())
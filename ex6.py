from utils import *
import sys
import scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.misc import imsave
from ex2reg import Ex2Reg
from scipy.optimize import minimize
from sklearn import svm
from functools import partial
from mpl_toolkits.mplot3d import Axes3D


class Ex6:
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

    def run(self):
        self.loading_and_visualizing_data()
        self.training_linear_svm()
        self.implementing_gaussian_kernel()
        self.visualizing_dataset_2()
        self.training_svm_with_rbf_kernel()
        self.visualizing_dataset_3()
        self.training_svm_with_rbf_kernel_dataset_3()

    @print_name
    # @pause_after
    def loading_and_visualizing_data(self):
        self.data = sio.loadmat("ex6-data/ex6data1.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.m, self.n = self.X.shape

        positive = self.y.flatten() == 1
        negative = ~positive

        fig, ax = plt.subplots()
        ax.plot(self.X[positive, 0], self.X[positive, 1], 'r+', markersize=7, label="y = 1")
        ax.plot(self.X[negative, 0], self.X[negative, 1], 'yo', markersize=7, label="y = 0")
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.savefig("ex6-data/plot_data.png")

    @print_name
    # @pause_after
    def training_linear_svm(self):
        C = 1
        model = svm.SVC(C=C, kernel='linear')
        model.fit(self.X, self.y.flatten())

        positive = self.y.flatten() == 1
        negative = ~positive
        b0 = model.intercept_
        b1, b2 = model.coef_.flatten()
        decision_boundary = -(b1 * self.X[:, 0] + b0)/b2

        fig, ax = plt.subplots()
        ax.plot(self.X[positive, 0], self.X[positive, 1], 'r+', markersize=7, label="y = 1")
        ax.plot(self.X[negative, 0], self.X[negative, 1], 'yo', markersize=7, label="y = 0")
        ax.plot(self.X[:, 0], decision_boundary, label="decision boundary")
        ax.legend()
        ax.set_title("C = {}".format(C))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.savefig("ex6-data/decision_boundary.png")

    @print_name
    # @pause_after
    def implementing_gaussian_kernel(self):
        x1 = np.array([1, 2, 1])
        x2 = np.array([0, 4, -1])
        sigma = 2
        sim = self.gaussian_kernel(x1, x2, sigma)

        print("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {} :".format(sigma))
        print("{:.6f}".format(sim))
        print("For sigma = 2, this value should be about 0.324652")

    def gaussian_kernel(self, x1, x2, sigma):
        d = x1 - x2
        return np.exp(-(d @ d.T)/(2 * sigma ** 2))

    @print_name
    # @pause_after
    def visualizing_dataset_2(self):
        self.data = sio.loadmat("ex6-data/ex6data2.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.m, self.n = self.X.shape

        positive = self.y.flatten() == 1
        negative = ~positive

        fig, ax = plt.subplots()
        ax.plot(self.X[positive, 0], self.X[positive, 1], 'r+', label="y = 1")
        ax.plot(self.X[negative, 0], self.X[negative, 1], 'yo', label="y = 0")
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.savefig("ex6-data/plot_data2.png")

    def training_svm_with_rbf_kernel(self):
        C = 1
        sigma = 0.1
        gamma = 1/(2 * sigma ** 2)
        model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(self.X, self.y.flatten())

        self.visualize_boundary(self.X, self.y, model, num=2)

    def visualize_boundary(self, X, y, model, num):
        positive = y.flatten() == 1
        negative = ~positive

        h = 0.02
        pad = 0.2
        x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
        y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.plot(X[positive, 0], X[positive, 1], 'r+', label="y = 1")
        ax.plot(X[negative, 0], X[negative, 1], 'yo', label="y = 0")
        ax.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.2)
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.savefig("ex6-data/decision_boundary{}.png".format(num))

    def visualizing_dataset_3(self):
        self.data = sio.loadmat("ex6-data/ex6data3.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.Xval = self.data['Xval']
        self.yval = self.data['yval']
        self.m, self.n = self.X.shape

        positive = self.y.flatten() == 1
        negative = ~positive

        fig, ax = plt.subplots()
        ax.plot(self.X[positive, 0], self.X[positive, 1], 'r+', label="y = 1")
        ax.plot(self.X[negative, 0], self.X[negative, 1], 'yo', label="y = 0")
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.savefig("ex6-data/plot_data3.png")

    def training_svm_with_rbf_kernel_dataset_3(self):
        C, sigma = self.cross_validation(self.X, self.y, self.Xval, self.yval)
        gamma = 1 / (2 * sigma ** 2)
        model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(self.X, self.y.flatten())

        self.visualize_boundary(self.X, self.y, model, num=3)

    def cross_validation(self, X, y, Xval, yval):
        values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        l = len(values)
        cv_error = np.zeros((l, l))

        for i in range(l):
            for j in range(l):
                C = values[i]
                sigma = values[j]
                gamma = 1 / (2 * sigma ** 2)
                model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
                model.fit(X, y.flatten())
                cv_error[i, j] = 1 - model.score(Xval, yval)
        i, j = np.unravel_index(np.argmin(cv_error), cv_error.shape)
        return values[i], values[j]


if __name__ == '__main__':
    ex6 = Ex6()
    sys.exit(ex6.run())
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


class Ex3:
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
        self.num_labels = None # Number of labels / classes
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

    def run(self):
        self.loading_and_visualizing_data()
        self.vectorize_logistic_regression()
        self.one_vs_all_training()

    @print_name
    # @pause_after
    def loading_and_visualizing_data(self):
        self.data = sio.loadmat("ex3-data/ex3data1.mat")
        self.X = self.data['X']
        self.y = self.data['y']
        self.m, self.n = self.X.shape
        self.num_labels = 10

        # 10x10 images sample matrix
        rand_indices = np.random.permutation(self.m)
        a = 10 # number of images on one row/column
        p = 20 # number of pixels in image row/column
        sample_matrix = np.vstack(np.hsplit(self.X[rand_indices[:a*a],:].reshape(-1, p).T, a))
        imsave("ex3-data/10x10_img_sample.png", sample_matrix)

    @print_name
    # @pause_after
    def vectorize_logistic_regression(self):
        theta_t = np.reshape([-2, -1, 1, 2], (-1, 1))
        X_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10))
        y_t = np.reshape([1, 0, 1, 0, 1], (-1, 1))
        l_t = 3
        cost = Ex2Reg.cost_function_reg(theta_t, X_t, y_t, l_t)[0,0]
        grad = Ex2Reg.gradient_reg(theta_t, X_t, y_t, l_t)
        print("Cost: {:.4f}".format(cost))
        print("Expected cost: 2.534819")
        print("Gradients:")
        print("[{:.4f} {:.4f} {:.4f} {:.4f}]".format(*grad))
        print("Expected gradients:")
        print("[0.146561 -0.548558 0.724722 1.398003]")

    @print_name
    # @pause_after
    def one_vs_all_training(self):
        self.l = 0.1
        self.theta = self.one_vs_all(self.X, self.y, self.num_labels, self.l)
        predicted = self.predict_one_vs_all(self.theta, self.X)
        print("Training set accuracy: {:.2f}%".format(np.mean(predicted == self.y.flatten()) * 100))

    @staticmethod
    def one_vs_all(X, y, num_labels, l):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        n += 1
        init_theta = np.zeros((n, 1))
        solver = 'L-BFGS-B'
        options = {'maxiter': 50}
        all_theta = np.zeros((num_labels, n))
        for c in range(1, num_labels + 1):
            res = minimize(fun=Ex2Reg.cost_function_reg,
                           x0=init_theta,
                           jac=Ex2Reg.gradient_reg,
                           args=(X, (y.flatten() == c).astype(int), l),
                           method=solver,
                           options=options)
            all_theta[c - 1] = res.x
            print("Class {:2.0f} cost: {:.4f}".format(c, res.fun))
        return all_theta

    @staticmethod
    def predict_one_vs_all(all_theta, X):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        p = sigmoid(X @ all_theta.T)
        return np.argmax(p, axis=1) + 1


if __name__ == '__main__':
    ex3 = Ex3()
    sys.exit(ex3.run())
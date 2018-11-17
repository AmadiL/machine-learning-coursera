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


class Ex3NN:
    def __init__(self):
        self.data = None # DataFrame from original data
        self.X = None # X matrix
        self.y = None # y vector
        self.l = None # Regularization parameter
        self.m = None # Number of training examples
        self.n = None # Number of features
        self.mu = None # Features' mean values
        self.sigma = None # Features' std dev values
        self.theta1 = None # Fitting parameters for layer 1
        self.theta2 = None # Fitting parameters for layer 2
        self.num_iters = None # Number of iterations
        self.num_labels = None # Number of labels / classes
        self.alpha = None # Alpha parameter
        self.J_history = None # Cost function history

        # Init

    def run(self):
        self.loading_and_visualizing_data()
        self.loading_parameters()
        self.implement_predict()

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
        imsave("ex3nn-data/10x10_img_sample.png", sample_matrix)

    @print_name
    # @pause_after
    def loading_parameters(self):
        weights = sio.loadmat("ex3nn-data/ex3weights.mat")
        self.theta1 = weights["Theta1"]
        self.theta2 = weights["Theta2"]

    @print_name
    # @pause_after
    def implement_predict(self):
        predicted = self.predict(self.theta1, self.theta2, self.X)
        print("Training set accuracy: {:.2f}%".format(np.mean(predicted == self.y.flatten()) * 100))

    def predict(self, theta1, theta2, X):
        m, _ = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        a1 = sigmoid(X @ theta1.T)
        a1 = np.hstack((np.ones((m, 1)), a1))
        p = np.argmax(sigmoid(a1 @ theta2.T), axis=1) + 1
        return p


if __name__ == '__main__':
    ex3nn = Ex3NN()
    sys.exit(ex3nn.run())
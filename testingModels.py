import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from reading import read, plotErrorBar

def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))

def lasso_regr(X, y):
    C = 5000
    model = Lasso(alpha=1 / (2 * C), max_iter=100000).fit(X, y)
    ypred = model.predict(X)

def lin_regr(X, y):
    C = 0.1
    model = linear_model.Ridge(alpha=1 / (2 * C)).fit(X, y)
    ypred = model.predict(X)

def kNN_regr(X, y):
    k=100
    gamma=3
    kernel = generate_gaussian_kernel_function(gamma)
    model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(X, y)
    ypred = model.predict(X)
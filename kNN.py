from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

from reading import read, plotErrorBar

def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))

def crossValidationK(X, y, ks):
    newks = []
    yMeanValues = []
    yVarianceValues = []
    fold = 5
    gamma = 1
    kernel = generate_gaussian_kernel_function(gamma)

    for k in ks:
        means = []
        mean = 0
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            #  kNN
            model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(X[train], y[train])
            ypred = model.predict(X[test])

            tmp = mean_squared_error(y[test], ypred)
            means.append(tmp)
            mean += tmp

        yMeanValues.append(mean / fold)

        # Calculate Variance
        sum = 0
        for i in means:
            tmp = i - mean
            sum += tmp * tmp
        sum = sum / (fold - 1)
        yVarianceValues.append(math.sqrt(sum))
        newks.append(float(k))

    npMeans = np.array(yMeanValues)
    npVar = np.array(yVarianceValues)
    plotErrorBar(newks, npMeans, npVar, 'k', 'K vs Mean - kNN', 'errorbarK.png')


def main():
    #   Get Training Data
    X, y = read('allKerryCars.csv')
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    crossValidationK(X, y, [1, 3, 5, 10, 20 ])

main()
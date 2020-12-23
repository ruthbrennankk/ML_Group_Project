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
    gamma = 3
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
    plotErrorBar(newks, npMeans, npVar, 'k', 'K vs Mean - kNN (gamma=3)', 'errorbarK.png')


def crossValidationG(X, y, gammas):
    newgs = []
    yMeanValues = []
    yVarianceValues = []
    fold = 5
    k = 100

    for gamma in gammas:
        means = []
        mean = 0
        kernel = generate_gaussian_kernel_function(gamma)
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            #  kNN
            model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(X[train], y[train])
            ypred = model.predict(X[test])
            #print(np.where(np.isnan(ypred)))
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
        newgs.append(float(gamma))

    npMeans = np.array(yMeanValues)
    npVar = np.array(yVarianceValues)
    plotErrorBar(newgs, npMeans, npVar, 'Gamma', 'Gamma vs Mean - kNN, k=100', 'errorbar_KNN_G_k=100.png')


def main():
    #   Get Training Data
    X, y = read('../Data/g_cars_final.csv')

    # Now you are all set to use these data to fit a KNN classifier.
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    crossValidationK(X, y, [1, 3, 5, 10, 20, 50, 100, 250, 500])
    crossValidationG(X, y, [0, 1, 3, 4, 5, 10, 25, 50, 100])

main()
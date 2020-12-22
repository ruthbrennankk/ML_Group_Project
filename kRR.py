import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model

from reading import read, plotErrorBar

def crossValidationC(X, y, Cs):
    newks = []
    yMeanValues = []
    yVarianceValues = []
    fold = 5
    gamma = 1

    for C in Cs:
        means = []
        mean = 0
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            model = linear_model.Lasso(alpha=1/C).fit(X[train], y[train])
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
        newks.append(float(C))

    npMeans = np.array(yMeanValues)
    npVar = np.array(yVarianceValues)
    plotErrorBar(newks, npMeans, npVar, 'C', 'C vs Mean - Lasso', 'errorbar.png')


def crossValidationG(X, y, gammas):
    newgs = []
    yMeanValues = []
    yVarianceValues = []
    fold = 5
    C = 0.1

    for gamma in gammas:
        means = []
        mean = 0
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):

            model = KernelRidge(alpha=1 / (2 * C), kernel='rbf', gamma=gamma).fit(X[train], y[train])
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
        newgs.append(float(gamma))

    npMeans = np.array(yMeanValues)
    npVar = np.array(yVarianceValues)
    plotErrorBar(newgs, npMeans, npVar, 'Gamma', 'Gamma vs Mean - KRR, C=0.1', 'errorbar_KNN_G_k=500.png')


def main():
    #   Get Training Data
    X, y = read('kerry_no_label.csv')

    # Now you are all set to use these data to fit a KNN classifier.
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    crossValidationC(X, y, [1, 3, 5, 10, 20])
    #crossValidationG(X, y, [0, 10, 25, 50, 100, 250, 500])

main()
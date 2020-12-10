import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

from reading import read, plotErrorBar

def addFeatures(X):
    poly = PolynomialFeatures(5)
    return poly.fit_transform(X)

def ridge(X, y, c):
    clf = linear_model.Ridge(alpha=1/(2*c))
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.intercept_)
    return clf, clf.coef_, clf.intercept_

def crossValidationK(X, y, cs):
    newcs = []
    yMeanValues = []
    yVarianceValues = []
    fold = 5

    for c in cs:
        means = []
        mean = 0
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            #   Ridge
            model = linear_model.Ridge(alpha=1 / (2 * c)).fit(X[train], y[train])
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
        newcs.append(float(c))

    npMeans = np.array(yMeanValues)
    npVar = np.array(yVarianceValues)
    plotErrorBar(newcs, npMeans, npVar, 'c', 'C vs Mean - Ridge', 'errorbar_RC.png')



def main():
    # Read in the data (using pandas)
    oldX,y = read('allKerryCars.csv')

    # equal to all combinations of powers of the two features up to power 5
    X = addFeatures(oldX)

    # Cross val for C
    crossValidationK(X, y, [1, 3, 5, 10, 20])


main()

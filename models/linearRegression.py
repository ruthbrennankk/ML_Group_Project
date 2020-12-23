import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

from models.assist.reading import read, plotErrorBar

def addFeatures(X, q):
    poly = PolynomialFeatures(q)
    return poly.fit_transform(X)

def ridge(X, y, c):
    clf = linear_model.Ridge(alpha=1/(2*c))
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.intercept_)
    return clf, clf.coef_, clf.intercept_

def crossValidationC(X, y, cs):
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

def crossValQ(oldX, y, qi_range):
    fold = 5
    mean_error = []
    std_error = []
    c = 10
    for qi in qi_range:
        print(qi)
        temp = []
        X = addFeatures(oldX, qi)
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            model = linear_model.Ridge(alpha=1 / (2 * c)).fit(X[train], y[train])
            ypred = model.predict(X[test])

            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(qi_range, mean_error, yerr=std_error)
    plt.title("Cross Val Polynomial Features (C=%d)" % (c))
    plt.xlabel('Q')
    plt.ylabel('Mean Square Error')
    #plt.savefig("errorbar_Ridge_q")
    plt.show()


def main():
    # Read in the data (using pandas)
    oldX,y = read('../Data Gathering/Data/g_cars_final.csv')

    # equal to all combinations of powers of the two features up to power 5
    #X = addFeatures(oldX,1)

    # Cross val for C
    crossValidationC(oldX, y, [0.01, 0.1, 1, 2, 5, 10, 100, 1000])

    #Cross val for polynomial features
    #crossValQ(oldX, y, [1,2,3]) #,4,5,6,7,8,9,10,11,12

main()

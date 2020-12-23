import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from reading import read, plotErrorBar

def add_poly_features(inputX, powIn):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(powIn).fit_transform(inputX)

def crossValidationC(X, y, Ci_range):
    #newcs = []
    mean_error = []
    std_error = []
    fold = 5

    for Ci in Ci_range:
        temp = []
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            #   Lasso
            model = Lasso(alpha=1/(2*Ci), max_iter=100000).fit(X[train], y[train])
            ypred = model.predict(X[test])
            # print(1/(2*Ci))
            # print(model.coef_)
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    #plotErrorBar(Ci_range, npMeans, npVar, 'c', 'C vs Mean - Ridge', 'errorbar_RC.png')
    plt.title("C vs Mean - LASSO")
    plt.rc('font',size=18)
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3, c='r')
    plt.xlabel('C')
    plt.ylabel('mean')
    plt.savefig("errorbar_Lasso_C")
    #plt.show()

def crossValQ(oldX, y, qi_range):
    fold = 5
    mean_error = []
    std_error = []
    C = 1000
    for qi in qi_range:
        temp = []
        X = add_poly_features(oldX, qi)
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            #   Lasso
            model = Lasso(alpha=1 / (2 * C)).fit(X[train], y[train])
            ypred = model.predict(X[test])

            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(qi_range, mean_error, yerr=std_error)
    plt.title("Cross Val Polynomial Features (C=%d)"%(C))
    plt.xlabel('Q')
    plt.ylabel('Mean Square Error')
    plt.savefig("errorbar_Lasso_q")
    #plt.show()


def main():
    # Read in the data (using pandas)
    oldX,y = read('g_cars_final.csv')

    # equal to all combinations of powers of the two features up to power 5
    # X = add_poly_features(oldX, 3)

    # Cross val for C
    #crossValidationC(oldX, y, [0.1, 1, 5, 10, 50,75, 100, 1000, 1500, 2000, 3000])
    crossValidationC(oldX, y, [1, 100, 1000, 5000, 7500, 10000, 15000, 20000])
    #Cross val for q (polynomials to add)
    #crossValQ(oldX, y, [1, 2, 3, 4, 5, 6, 7, 8])

main()
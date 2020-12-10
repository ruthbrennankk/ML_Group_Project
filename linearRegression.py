import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib  import cm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

def normalise(arr) :
    # Find the mean
    m = np.mean(arr)
    # Find standard deviation
    s = np.amax(arr) - np.amin(arr) #np.std(arr)
    # normalise data using z score
    return (arr - m) / s

def read(filename):
    df = pd.read_csv(filename, comment='#', header=None)
    print(df.head())

    X1 = normalise(df.iloc[:, 0]) # Brands
    X2 = normalise(df.iloc[:, 1]) # model
    X3 = normalise(df.iloc[:, 2]) # transmission
    X4 = normalise(df.iloc[:, 3]) # Colour
    X5 = normalise(df.iloc[:, 4]) # mileage
    X6 = normalise(df.iloc[:, 5]) # year
    X = np.column_stack((X1,X2,X3,X4,X5,X6))
    y = normalise(df.iloc[:,6])

    return (X, y)

def addFeatures(X):
    poly = PolynomialFeatures(5)
    return poly.fit_transform(X)

def ridge(X, y, c):
    clf = linear_model.Ridge(alpha=1/(2*c))
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.intercept_)
    return clf, clf.coef_, clf.intercept_


def plotErrorBar(x, mean, var, xlabel, title, image):
    # create an index for each tick position
    xi = list(range(len(x)))
    npc = np.array(xi)

    fig, ax = plt.subplots()
    plt.errorbar(npc, mean, var, linewidth=3, marker='o', c='b')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean')
    plt.xticks(xi, x)
    fig.show()
    #fig.savefig(image)

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

    # (i)(b) - add extra polynomial features
    # equal to all combinations of powers of the two features up to power 5
    X = addFeatures(oldX)
    #print(X)

    # Cross val for C
    crossValidationK(X, y, [1, 3, 5, 10, 20])



main()

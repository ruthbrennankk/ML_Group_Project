import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))


def read(filename):
    df = pd.read_csv(filename, comment='#', header=None)
    print(df.head())

    X1 = normalise(df.iloc[:, 1]) # Brands
    X2 = normalise(df.iloc[:, 2]) # model
    X3 = normalise(df.iloc[:, 3]) # transmission
    X4 = normalise(df.iloc[:, 4]) # Colour
    X5 = normalise(df.iloc[:, 5]) # mileage
    X6 = normalise(df.iloc[:, 6]) # year
    X7 = normalise(df.iloc[:, 7]) # seats
    X8 = normalise(df.iloc[:, 8]) # doors
    X = np.column_stack((X1, X2, X3, X4, X5, X6, X8))
    y = np.array(df.iloc[:,9])

    return (X, y)

def normalise(arr) :
    # Find the mean
    m = np.mean(arr)
    # Find standard deviation
    s = np.amax(arr) - np.amin(arr) #np.std(arr)
    # normalise data using z score
    return (arr - m) / s

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
            #   kNN
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
    plotErrorBar(newks, npMeans, npVar, 'k', 'K vs Mean (with Variance)', 'errorbarK.png')


def main():
    #   Get Training Data
    X, y = read('car.csv')
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    crossValidationK(X, y, [1, 3, 5, 10, 20 ])

main()
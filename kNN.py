import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))


def read(filename):
    df = pd.read_csv(filename, comment='#', header=None)
    print(df.head())

    X1 = df.iloc[:, 1] # Brands
    X2 = df.iloc[:, 2] # model
    X3 = df.iloc[:, 3] # transmission
    X4 = df.iloc[:, 4] # Colour
    X5 = df.iloc[:, 5] # mileage
    X6 = df.iloc[:, 6] # year
    X7 = df.iloc[:, 7] # seats
    X8 = df.iloc[:, 8] # doors
    X = np.column_stack((X1, X2, X3, X4, X5, X6, X8))
    y = np.array(df.iloc[:,9])

    return (X, y)


def part_a():
    #   Get Training Data
    X, y = read('car.csv')

    m = 10

    # Generate test data
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    # Loop, predict & plot with gammas
    gammas = [0, 1, 5, 10, 25]
    for gamma in gammas:
        kernel = generate_gaussian_kernel_function(gamma)
        #   kNN
        model = KNeighborsRegressor(n_neighbors=m, weights=kernel).fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)

        # Plot
        plt.rc('font', size=18);
        # plt.rcParams['figure.constrained_layout.use'] = True
        plt.scatter(Xtrain[:,1], ytrain, color='red', marker='+')  # Plot training data
        plt.plot(Xtest[:,1], ypred, color='blue')  # Plot predictions
        plt.title('KNeighborsRegressor_Gamma_' + str(gamma))
        plt.xlabel("input x");
        plt.ylabel("output y")
        plt.legend(["kNN", "train"])
        plt.show()
        #plt.savefig('ii_KNeighborsRegressor_Gamma_' + str(gamma))
        plt.clf()

part_a()
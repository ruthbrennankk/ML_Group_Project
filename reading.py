import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read(filename):
    df = pd.read_csv(filename, comment='#', header=None)
    print(df.head())

    X1 = normalise(df.iloc[:, 0])  # Brands
    X2 = normalise(df.iloc[:, 1])  # model
    X3 = normalise(df.iloc[:, 2])  # transmission
    X4 = normalise(df.iloc[:, 3])  # Colour
    X5 = normalise(df.iloc[:, 4])  # mileage
    X6 = normalise(df.iloc[:, 5])  # year

    # X = np.column_stack((X1,X2,X3,X4,X5,X6))  # Everything
    # X = np.column_stack((X2,X3,X4,X5,X6))  # Not Brand
    # X = np.column_stack((X1, X3, X4, X5, X6))  # Not Model
    X = np.column_stack((X1, X2, X3, X5, X6))  # Not Colour

    y = normalise(df.iloc[:, 6])

    return (X, y)

def readNN(filename):
    df = pd.read_csv(filename, comment='#', header=None)
    print(df.head())

    X1 = (df.iloc[:, 0])  # Brands
    X2 = (df.iloc[:, 1])  # model
    X3 = (df.iloc[:, 2])  # transmission
    X4 = (df.iloc[:, 3])  # Colour
    X5 = (df.iloc[:, 4])  # mileage
    X6 = (df.iloc[:, 5])  # year

    # X = np.column_stack((X1,X2,X3,X4,X5,X6))  # Everything
    # X = np.column_stack((X2,X3,X4,X5,X6))  # Not Brand
    # X = np.column_stack((X1, X3, X4, X5, X6))  # Not Model
    X = np.column_stack((X1, X2, X3, X5, X6))  # Not Colour

    y = (df.iloc[:, 6])

    return (X, y)

def normalise(arr) :
    # Find the mean
    m = np.mean(arr)
    # Find standard deviation
    s = np.amax(arr) - np.amin(arr) #np.std(arr) for between 0 and 1
    # normalise data using z score
    return (arr - m) / s

def plotErrorBar(x, mean, var, xlabel, title, image):
    # create an index for each tick position
    xi = list(range(len(x)))
    npc = np.array(xi)

    fig, ax = plt.subplots()
    plt.errorbar(npc, mean, var, linewidth=3, marker='o', c='r')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean')
    plt.xticks(xi, x)
    #fig.show()
    fig.savefig(image)
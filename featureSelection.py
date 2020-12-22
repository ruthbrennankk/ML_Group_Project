import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kNN import generate_gaussian_kernel_function
from reading import encode_df
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

def linear(Xtrain, ytrain, Xtest, ytest):
    model = LinearRegression().fit(Xtrain, ytrain)

def Ridge(Xtrain, ytrain, Xtest, ytest):
    c = 1
    model = Ridge(alpha=1 / (2 * c)).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

def kNN(Xtrain, ytrain, Xtest, ytest):
    k=10
    kernel = generate_gaussian_kernel_function(1)
    #  kNN
    model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

def main():
    # Import small dataset
    #df = pd.read_csv('small.csv', comment='#')
    df = encode_df('small.csv')


    # Split it into 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


    # Linear
    # Ridge
    # Lasso
    # kernalized kNN

    # Get test percentage, mean square error and plot predictions

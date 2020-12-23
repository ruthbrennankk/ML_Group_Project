import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reading import encode_df, read_df
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from testingModels import MSE, MAE, r_sq, lasso_regr, lin_regr, kNN_regr

def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))

def linear(Xtrain, ytrain, Xtest):
    model = LinearRegression().fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return ypred

def Ridge(Xtrain, ytrain, Xtest, ytest):
    c = 1
    model = Ridge(alpha=1 / (2 * c)).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return ypred

def kNN(Xtrain, ytrain, Xtest, ytest):
    k=10
    kernel = generate_gaussian_kernel_function(1)
    #  kNN
    model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return ypred

def encode(df, columnNames):
    for c in columnNames :
        # use pd.concat to join the new columns with your original dataframe
        df = pd.concat([df, pd.get_dummies(df[c], prefix=c)],axis=1)
        # now drop the original 'country' column (you don't need it anymore)
        df.drop([c], axis=1, inplace=True)

    price = df['Price']
    df.drop(['Price'], axis=1, inplace=True)
    df = pd.concat([df, price], axis=1)
    return df

def drop(df, columnNames) :
    for c in columnNames :
        # now drop the original 'country' column (you don't need it anymore)
        df.drop([c], axis=1, inplace=True)

    price = df['Price']
    df.drop(['Price'], axis=1, inplace=True)
    df = pd.concat([df, price], axis=1)
    return df

def main():
    # Import small dataset
    #df = pd.read_csv('small.csv', comment='#')
    df = read_df('small.csv')
    print(df.head())

    # Test with all features
    everything_df = encode(df, ['Brands', 'Models', 'Colour'])
    X = everything_df.iloc[:, :-1].to_numpy()
    y = (everything_df.iloc[:, -1:].to_numpy())
    # Split it into 80/20
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)

    lasso_regr(Xtrain, Xtest, ytrain, ytest)
    lin_regr(Xtrain, Xtest, ytrain, ytest)
    kNN_regr(Xtrain, Xtest, ytrain, ytest)



main()
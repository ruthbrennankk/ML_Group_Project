import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from models.assist.reading import read


def generate_gaussian_kernel_function(gamma):
    weights = lambda dists: np.exp(-gamma * (dists ** 2))
    return lambda dists: weights(dists) / np.sum(weights(dists))

def MAE(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    mae_train = mean_absolute_error(ytrain, ypred)
    ypred = model.predict(Xtest)
    mae_test = mean_absolute_error(ytest, ypred)
    print("MAE    Train:%f      Test:%f"%(mae_train, mae_test))
    #print("The MAE for the %s model with the training data was %f and with the test data was %f"%(name, mae_train, mae_test))

def MSE(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    mse_train = mean_squared_error(ytrain, ypred, squared=True)
    ypred = model.predict(Xtest)
    mse_test = mean_squared_error(ytest, ypred, squared=True)
    print("MSE    Train:%f      Test:%f" % (mse_train, mse_test))
    #print("The RMSE for the %s model with the training data was %f and with the test data was %f" % (name, mse_train, mse_test))

def r_sq(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    rsq_train = r2_score(ytrain, ypred)
    ypred = model.predict(Xtest)
    rsq_test = r2_score(ytest, ypred)
    print("RSQ    Train:%f      Test:%f" % (rsq_train, rsq_test))
    #print("The R squared score for the %s model with the training data was %f and with the test data was %f" % (name, rsq_train, rsq_test))

def plot(name, model, Xtrain, Xtest, ytrain, ytest, filename):
    ypred = model.predict(Xtrain)
    fig, axs = plt.subplots()
    axs.scatter(Xtrain[:,1], ytrain, s=20, c='b', marker='+')
    axs.scatter(Xtrain[:,1], ypred, s=20, c='r', marker='+')
    axs.set_title(name + ' - Precicted Training Price vs Mileage')
    axs.set_xlabel('Mileage (km)')
    axs.set_ylabel('Price (€)')
    axs.legend(['Target', 'Predictions'])
    fig.show()
    #fig.savefig('Plots/Prediction Plots/' + filename + '_train_pred')

    ypred = model.predict(Xtest)
    fig, axs = plt.subplots()
    axs.scatter(Xtest[:, 1], ytest, s=20, c='b', marker='+')
    axs.scatter(Xtest[:, 1], ypred, s=20, c='r', marker='+')
    axs.set_title(name + ' - Predicted Test Price vs Mileage')
    axs.set_xlabel('Mileage (km)')
    axs.set_ylabel('Price (€)')
    axs.legend(['Target', 'Predictions'])
    fig.show()
    #fig.savefig('Plots/Prediction Plots/' + filename + '_test_pred')

def lasso_regr(Xtrain, Xtest, ytrain, ytest):
    C = 20000
    model = Lasso(alpha=1 / (2 * C), max_iter=100000).fit(Xtrain, ytrain)
    print("LASSO:")
    MAE("lasso", model, Xtrain, Xtest, ytrain, ytest)
    MSE("lasso", model, Xtrain, Xtest, ytrain, ytest)
    r_sq("lasso", model, Xtrain, Xtest, ytrain, ytest)
    plot("Lasso Regression", model, Xtrain, Xtest, ytrain, ytest, "lr")

def lin_regr(Xtrain, Xtest, ytrain, ytest):
    C = 1
    model = linear_model.Ridge(alpha=1 / (2 * C)).fit(Xtrain, ytrain)
    print("Ridge:")
    MAE("ridge", model, Xtrain, Xtest, ytrain, ytest)
    MSE("ridge", model, Xtrain, Xtest, ytrain, ytest)
    r_sq("ridge", model, Xtrain, Xtest, ytrain, ytest)
    plot("Ridge Regression", model, Xtrain, Xtest, ytrain, ytest, "r")

def kNN_regr(Xtrain, Xtest, ytrain, ytest):
    k=50
    gamma=100
    kernel = generate_gaussian_kernel_function(gamma)
    model = KNeighborsRegressor(n_neighbors=k, weights=kernel).fit(Xtrain, ytrain)
    MAE("kNN", model, Xtrain, Xtest, ytrain, ytest)
    MSE("kNN", model, Xtrain, Xtest, ytrain, ytest)
    r_sq("ridge", model, Xtrain, Xtest, ytrain, ytest)
    plot("kNN Regression", model, Xtrain, Xtest, ytrain, ytest, "knn")

def dummy_regr(Xtrain, Xtest, ytrain, ytest):
    strategy = "mean"
    model = DummyRegressor(strategy=strategy).fit(Xtrain, ytrain)
    print("Dummy:")
    MAE("dummy", model, Xtrain, Xtest, ytrain, ytest)
    MSE("dummy", model, Xtrain, Xtest, ytrain, ytest)
    r_sq("dummy", model, Xtrain, Xtest, ytrain, ytest)
    plot("Dummy Regression", model, Xtrain, Xtest, ytrain, ytest, "d")

def main():
    #Load data and hold back a test sample of size 0.2
    X, y = read("Data Gathering/Data/g_cars_final.csv")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2)
    lasso_regr(Xtrain, Xtest, ytrain, ytest)
    lin_regr(Xtrain, Xtest, ytrain, ytest)
    kNN_regr(Xtrain, Xtest, ytrain, ytest)
    dummy_regr(Xtrain, Xtest, ytrain, ytest)

main();

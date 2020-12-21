import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from reading import read, plotErrorBar


def main():
    # Read in the data (using pandas)
    # X, y = read('allKerryCars.csv')

    #their data set
    from sklearn.datasets import load_boston
    boston = load_boston()
    print(boston.keys())
    print(boston.data.shape)
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['PRICE'] = boston.target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    data_dmatrix = xgb.DMatrix(data=X, label=y)

    import xgboost as xgb
    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    model.fit(x_train, y_train)
    model.score(x_test, y_test)
    0.82702702702702702


main()
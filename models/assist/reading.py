import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read(filename):
    # df = pd.read_csv(filename, comment='#', header=None)

    df = encode_df(filename)
    print(df.head())

    # X = df.iloc[:,:-1].to_numpy()
    # y = normalise(df.iloc[:,-1:].to_numpy())

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1:].to_numpy()

    return (X, y)

def encode_df(filename):
    # og_df = pd.read_csv('kerry_no_label.csv', comment='#')
    og_df = pd.read_csv(filename, comment='#')
    print(og_df.head())

    # df now has two columns: name and country
    df = pd.DataFrame({
            'Brands': og_df.iloc[:, 1],
            'Models': og_df.iloc[:, 2],
            'Transmission': og_df.iloc[:, 3],
            #'Colour': og_df.iloc[:, 4],
            'Mileage': normalise(og_df.iloc[:, 5]),
            'Year': normalise(og_df.iloc[:, 6]),
            'Price' : normalise(og_df.iloc[:,7])
        })
    print(df.head())

    # Brand
    # use pd.concat to join the new columns with your original dataframe
    df = pd.concat([df, pd.get_dummies(df['Brands'], prefix='brand')],axis=1)
    # now drop the original 'country' column (you don't need it anymore)
    df.drop(['Brands'], axis=1, inplace=True)

    #Model
    # use pd.concat to join the new columns with your original dataframe
    df = pd.concat([df, pd.get_dummies(df['Models'], prefix='model')],axis=1)
    # now drop the original 'country' column (you don't need it anymore)
    df.drop(['Models'], axis=1, inplace=True)

    # #Colour
    # # use pd.concat to join the new columns with your original dataframe
    # df = pd.concat([df, pd.get_dummies(df['Colour'], prefix='colour')],axis=1)
    # # now drop the original 'country' column (you don't need it anymore)
    # df.drop(['Colour'], axis=1, inplace=True)

    price = df['Price']
    df.drop(['Price'], axis=1, inplace=True)
    df = pd.concat([df, price], axis=1)

    # print(df.head())
    return df

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
    fig.show()
    #fig.savefig(image)

def read_df(filename):
    # og_df = pd.read_csv('kerry_no_label.csv', comment='#')
    og_df = pd.read_csv(filename, comment='#')
    # print(og_df.head())

    # df now has two columns: name and country
    df = pd.DataFrame({
            'Brands': og_df.iloc[:, 1],
            'Models': og_df.iloc[:, 2],
            'Transmission': og_df.iloc[:, 3],
            'Colour': og_df.iloc[:, 4],
            'Mileage': (og_df.iloc[:, 5]),
            'Year': (og_df.iloc[:, 6]),
            'Price': (og_df.iloc[:, 7])
        })
    # print(df.head())

    # X = df.iloc[:,:-1].to_numpy()
    # y = (df.iloc[:,-1:].to_numpy())
    # return (X, y)
    return df

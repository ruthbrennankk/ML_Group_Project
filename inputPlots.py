import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reading import read_df

def inputScatter():
    X,y = read_df('g_cars_final.csv')

    # Brand
    fig, axs = plt.subplots()
    axs.scatter( X[:,0], y, s=20, c='r', marker='+')
    axs.set_title('Price vs Brand')
    plt.xticks(rotation=90)
    axs.set_xlabel('Brand')
    axs.set_ylabel('Price (€)')
    # fig.show()
    fig.savefig('input_scatter_brand')

    # # Model
    # fig, axs = plt.subplots()
    # axs.scatter( X[:, 1], y, s=20, c='b', marker='+')
    # #plt.xticks(rotation=90)
    # fig.autofmt_xdate()  # make space for and rotate the x-axis tick labels
    # axs.set_title('Price vs Model')
    # axs.set_xlabel('Model')
    # axs.set_ylabel('Price (€)')
    # fig.show()

    # Transmission
    fig, axs = plt.subplots()
    axs.scatter( X[:, 2], y, s=20, c='g', marker='+')
    axs.set_xticks(range(0, 2))
    axs.set_xticklabels(['Manual', 'Automatic'])
    axs.set_title('Price vs Transmission')
    axs.set_xlabel('Transmission')
    axs.set_ylabel('Price (€)')
    # fig.show()
    fig.savefig('input_scatter_transmission')

    # Mileage
    fig, axs = plt.subplots()
    axs.scatter( X[:, 4], y, s=20, c='y', marker='+')
    axs.set_title('Price vs Mileage')
    axs.set_xlabel('Mileage (km)')
    axs.set_ylabel('Price (€)')
    # fig.show()
    fig.savefig('input_scatter_mileage')

    # Age
    fig, axs = plt.subplots()
    axs.scatter( X[:, 5], y, s=20, c='m', marker='+')
    axs.set_xticks(range(0, 20))
    axs.set_title('Price vs Age')
    axs.set_xlabel('Age (Years)')
    axs.set_ylabel('Price (€)')
    # fig.show()
    fig.savefig('input_scatter_age')



def inputFreq():
    X, y = read_df('g_cars_final.csv')

    # Brand
    fig, axs = plt.subplots()
    unique_elements, counts_elements = np.unique(X[:,0], return_counts=True)
    axs.bar(unique_elements, counts_elements)
    axs.set_title('Brand Frequency')
    axs.set_xlabel('Brands')
    axs.set_ylabel('Frequency')
    plt.xticks(rotation=90)
    # fig.show()
    fig.savefig('input_bar_brand')

    # # Model
    # fig, axs = plt.subplots()
    # unique_elements, counts_elements = np.unique(X[:,1], return_counts=True)
    # axs[1].ax.bar(unique_elements, counts_elements, c='b')
    # plt.xticks(rotation=90)
    # axs[1].set_title('Model Frequency')
    # axs[1].set_xlabel('Unique Models')
    # axs[1].set_ylabel('Frequency')

    # Transmission
    fig, axs = plt.subplots()
    unique_elements, counts_elements = np.unique(X[:,2], return_counts=True)

    if unique_elements[0]==1 :
        x_labels = ['Automatic', 'Manual']
    else :
        x_labels = ['Manual', 'Automatic']

    axs.bar(x_labels, counts_elements)
    axs.set_title('Transmission Frequency')
    axs.set_xlabel('Transmissions')
    axs.set_ylabel('Frequency')
    # fig.show()
    fig.savefig('input_bar_transmission')

    # Mileage
    fig, axs = plt.subplots()
    unique_elements, counts_elements = np.unique( X[:,4], return_counts=True)

    labels = ['<50,000', '50,000 - 99,999', '100,000 - 149,999', '150,000 - 199,999', '200,000 - 249,999', '250,000 - 299,999', '300,000 - 349,999', '350,000 - 400,000','>400,000' ]
    count = [0,0,0,0,0,0,0,0,0]
    for i in range(len(unique_elements)) :
        if unique_elements[i] < 50000 :
             count[0] = count[0] + counts_elements[i]
        elif unique_elements[i] < 100000 :
            count[1] = count[1] + counts_elements[i]
        elif unique_elements[i] < 150000 :
            count[2] = count[2] + counts_elements[i]
        elif unique_elements[i] < 200000 :
            count[3] = count[3] + counts_elements[i]
        elif unique_elements[i] < 250000 :
            count[4] = count[4] + counts_elements[i]
        elif unique_elements[i] < 300000 :
            count[5] = count[5] + counts_elements[i]
        elif unique_elements[i] < 350000 :
            count[6] = count[6] + counts_elements[i]
        elif unique_elements[i] < 400000 :
            count[7] = count[7] + counts_elements[i]
        else :
            count[8] = count[8] + counts_elements[i]


    axs.bar(labels, count)
    fig.autofmt_xdate()  # make space for and rotate the x-axis tick labels
    axs.set_title('Mileage Frequency')
    axs.set_xlabel('Mileage')
    axs.set_ylabel('Frequency')
    # fig.show()
    fig.savefig('input_bar_mileage')

    # Age
    fig, axs = plt.subplots()
    unique_elements, counts_elements = np.unique(X[:,5], return_counts=True)
    axs.bar(unique_elements, counts_elements)
    axs.set_title('Age Frequency')
    axs.set_xlabel('Ages')
    axs.set_ylabel('Frequency')
    # fig.show()
    fig.savefig('input_bar_age')


inputFreq()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from week_4 import a, b, c

def lr(X,y, order, c) :
    print('\nLR')
    X = a.addFeatures(X, order)  # add new features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split data
    truePos, trueNeg = a.countNegPos(y_test)  # count for confusion martrix

    model = a.logReg(X_train, y_train, X_test, y_test, c)  # get predictions

    predictions = model.predict(X_test)  # get predictions
    predPos, predNeg = a.countNegPos(predictions)  # count for confusion matrix
    print ('\ntruePos = '+str(truePos))
    print('trueNeg = ' + str(trueNeg))
    print ('\npredPos = '+str(predPos))
    print('predNeg = ' + str(predNeg))

    return X_test, y_test, model

def kNN(X, y, k):
    print('\nkNN')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    truePos, trueNeg = b.countNegPos(y_test)

    model = b.kNN(X_train, y_train, X_test, y_test,  k)
    predictions = model.predict(X_test)
    predPos, predNeg = b.countNegPos(predictions)
    print ('\ntruePos = '+str(truePos))
    print('trueNeg = ' + str(trueNeg))
    print ('\npredPos = '+str(predPos))
    print('predNeg = ' + str(predNeg))

    return X_test, y_test, model


def ROC(X, y):


    plt.rc('font', size = 18); #plt.rcParams['figure.constrained_layout.use'] = True

    #LR
    #X_test, y_test, model = lr(X,y, 10, 100) #(i)
    X_test, y_test, model = lr(X, y, 7, 100)  # (ii)
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
    roc_auc = auc(fpr, tpr)
    print('LR AUC = ' + str(roc_auc))
    plt.plot(fpr, tpr)

    #kNN
    #X_test, y_test, model = kNN(X, y, 3)   #(i)
    X_test, y_test, model = kNN(X, y, 5)    #(ii)
    y_scores = model.predict_proba(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_scores[:,1])
    roc_auc = auc(fpr, tpr)
    print('kNN AUC = ' + str(roc_auc))
    plt.plot(fpr, tpr)

    #Baseline
    plt.plot(0, 1, 'ro')
    c.main()

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], 'g--')
    plt.title('Receiver Operating Characteristic - ROC')
    plt.legend(['LR', 'kNN', 'Baseline'])
    #plt.show()
    plt.savefig('ROC.png')

def main():
    dataset1 = "week_4.csv"
    dataset2 = "week_4_d2.csv"
    X, y = a.read(dataset2)
    ROC(X, y)

main()
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from hw9 import Dtrain_Dtest


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def p1():
    x = np.array([[2], [1]])
    y = np.array([[-1]])
    w1 = np.full((2, 2), 0.1501)
    w2 = np.full((2, 1), 0.1501)

    # tanh
    s1 = np.dot(w1.T, x)
    x1 = np.tanh(s1)

    # identity
    s2 = np.dot(w2.T, x1)
    x2 = s2

    # error, one point
    Ein = (1 / 4) * np.sum((x2 - y) ** 2)

    delta2Identity = 2 * (x2 - y)
    w2Identity = (1 / 2) * np.dot(x1, delta2Identity.T)
    delta2TanH = 2 * (np.tanh(s2) - y) * 1 - np.tanh(s2) ** 2
    w2TanH = (1 / 2) * np.dot(x1, delta2TanH.T)

    delta1Identity = np.dot(w2, delta2Identity) * (1 - np.tanh(s1) ** 2)
    w1Identity = (1 / 2) * np.dot(x, delta1Identity.T)
    delta1TanH = np.dot(w2, delta2TanH) * (1 - np.tanh(s1) ** 2)
    w1TanH = (1 / 2) * np.dot(x, delta1TanH.T)

    print("identity:\nw1: {}\nw2: {}".format(w1Identity, w2Identity))
    print("tanh:\nw1: {}\nw2: {}".format(w1TanH, w2TanH))
    print(Ein)


def p2():
    pass


def p4CV(y1, y2):
    if len(y1) != len(y2):
        print("not same length")
        return 0

    err = 0
    for i in range(len(y1)):
        if y1[i] != y2[i]:
            err += 1
    return err / len(y1)


def p4a(trainingData ,regularizer, scale, step, testingData=None):
    X, y = switchData(trainingData)
    SVM = SVC(kernel="poly", degree=8, C=regularizer)
    SVM.fit(X, y)

    x1s = np.arange(-scale, scale, step)
    x2s = np.arange(-scale, scale, step)

    # # uncomment for problem 4(a) # # #
    newX = []
    for x1 in x1s:
        for x2 in x2s:
            newX.append((x1, x2))

    outputs = SVM.predict(newX)
    i = 0
    for x1 in x1s:
        for x2 in x2s:
            Color = (0.68, 0.85, 0.90) if outputs[i] == 1 else (1.0, 0.71, 0.76)
            plt.plot(x1, x2, marker="o", color=Color)
            i += 1

    if testingData:
        Xtest, yTest = switchData(testingData)
        for i in range(len(Xtest)):
            Color = "blue" if yTest[i] == 1 else "red"
            Label = "o" if yTest[i] == 1 else "x"
            plt.plot(Xtest[i][0], Xtest[i][1], marker=Label, color=Color)
    else:
        for i in range(len(X)):
            Color = "blue" if y[i] == 1 else "red"
            Label = "o" if y[i] == 1 else "x"
            plt.plot(X[i][0], X[i][1], marker=Label, color=Color)

    plt.show()


def p4c(trainingData, testingData):
    X, y = switchData(trainingData)
    Xtest, yTest = switchData(testingData)
    i = 0.01
    lowestErr = 1
    lowestC = 0
    while i <= 10:
        SVM = SVC(kernel="poly", degree=8, C=i)
        SVM.fit(X, y)
        outputs = SVM.predict(Xtest)
        currErr = p4CV(yTest, outputs)
        if currErr < lowestErr:
            lowestErr = currErr
            lowestC = i
        i += 0.01

    print("lowestC: {}, lowestErr: {}".format(lowestC, lowestErr))
    p4a(trainingData, lowestC, 1, 0.05, testingData)



def p4(trainingData, testingData, regularizer=0.01, scale=1, step=0.05):
    # p4a(trainingData, regularizer, scale, step)
    p4c(trainingData, testingData)


def switchData(data):
    X = [(d[0], d[1]) for d in data]
    y = [d[2] for d in data]
    return X, y


if "__main__" == __name__:
    # p1()
    file = "ZipDigits.all"
    trainingData, testingData = Dtrain_Dtest(file)
    p4(trainingData, testingData, regularizer=10, step=0.05)

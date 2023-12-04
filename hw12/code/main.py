import numpy as np
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


if "__main__" == __name__:
    p1()
    file = "ZipDigits.all"
    data = Dtrain_Dtest(file)
    trainingData = data[0]
    testingData = data[1]

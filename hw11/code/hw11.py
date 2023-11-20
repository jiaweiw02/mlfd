import copy
import time
from branchBound import cluster
import matplotlib.pyplot as plt
import numpy as np
from knn import knn, plotDecisionRegion, distance


def generateData(filename):
    train = open(filename, 'r')
    allDigits = []
    for line in train:
        newData = [float(0) for i in range(257)]
        data = line.strip().split(" ")
        newData[0] = float(data[0])
        for i in range(1, len(data)):
            newData[i] = float(data[i])
        allDigits.append(newData)

    return allDigits


def computeIntensity(data):
    intensities = []
    minH, minV, maxH, maxV = np.inf, np.inf, -1, -1
    for d in data:
        dig = d[0]
        formatData = []
        for i in range(1, len(d), 16):
            formatData.append(d[i:i + 16])

        horIntensity = 0
        verIntensity = 0

        for i in range(8):
            for j in range(16):
                horIntensity += abs(formatData[i][j] - formatData[15 - i][j])

        for i in range(16):
            for j in range(8):
                verIntensity += abs(formatData[i][j] - formatData[i][15 - j])

        minH, maxH = min(minH, horIntensity), max(maxH, horIntensity)
        minV, maxV = min(minV, verIntensity), max(maxV, verIntensity)

        intensities.append(([horIntensity, verIntensity], 1 if dig == 1 else -1))
    return intensities, (minH, maxH), (minV, maxV)


def normalize(d, H, V):
    minH, maxH, minV, maxV = H[0], H[1], V[0], V[1]

    HScale, HShift = 2.0 / (maxH - minH), (minH + maxH) / 2.0
    VScale, VShift = 2.0 / (maxV - minV), (minV + maxV) / 2.0

    res = []
    for point in d:
        x1 = (point[0][0] - HShift) * HScale
        x2 = (point[0][1] - VShift) * VScale
        res.append(([x1, x2], point[1]))
    return res


def choose300(data):
    chosen = set()
    Dtrain = []
    Dtest = []

    iterations = 0
    while iterations < 300:
        rand = np.random.randint(0, len(data))
        if rand in chosen:
            continue
        Dtrain.append(data[rand])
        iterations += 1

    for i in range(len(data)):
        if i in chosen:
            continue
        Dtest.append(data[i])

    return Dtrain, Dtest


def Dtrain_Dtest(filename):
    d = computeIntensity(generateData(filename))
    normalizeD = normalize(d[0], d[1], d[2])
    res = choose300(normalizeD)
    return res[0], res[1]


def crossValidationNN(k, dTrain):
    err = 0
    for p in dTrain:
        new_dTrain = copy.deepcopy(dTrain)
        new_dTrain.remove(p)
        if knn(k, new_dTrain, p[0]) != p[1]:
            err += 1
    return err / len(dTrain)


def gaussianKernel(z):
    exp = -1 / 2 * (z ** 2)
    return np.e ** exp


def pseudoInverse(Z, y, reg):
    ZT = np.transpose(Z)
    ZTZ = np.dot(ZT, Z) + np.dot(np.identity(len(Z[0])), reg)
    ZTZ_INVERSE = np.linalg.inv(ZTZ)
    ZTZ_INVERSE_ZT = np.dot(ZTZ_INVERSE, ZT)
    return np.dot(ZTZ_INVERSE_ZT, y)


def errorClassification(w, points, y):
    # calculate E_IN
    E = 0
    for i in range(len(points)):
        x = points[i]
        output = y[i]
        if np.sign(np.dot(w, x)) != output:
            E += 1
    return E / len(points)


def crossValidationRBF(k, dTrain, reg=0):
    noOutput = [x for x, y in dTrain]
    r = 2 / np.sqrt(k)
    centers = cluster(k, noOutput)
    Z = []
    Y = []

    for x, y in dTrain:
        tmpZ = [1] + [gaussianKernel(distance(c, x) / r) for c, _, _ in centers]
        Z.append(tmpZ)
        Y.append(y)
    wStar = pseudoInverse(Z, Y, reg)
    return errorClassification(wStar, Z, Y), wStar, [c for c, _, _ in centers]


def errorNN(k, dTrain, dVal):
    err = 0
    for p in dVal:
        if knn(k, dTrain, p[0]) != p[1]:
            err += 1
    return err / len(dVal)


def plotCV(dTrain, NN=True, maxK=150):
    currTime = time.time()
    lowest = 1
    lowestK = 0

    kX = range(1, maxK + 1)
    kY = []
    W = None
    centers = None
    for k in range(1, maxK + 1):
        curr = None
        if NN:
            curr = crossValidationNN(k, dTrain)
        else:
            curr, W, centers = crossValidationRBF(k, dTrain, 0)
        kY.append(curr)
        if curr < lowest:
            lowest = curr
            lowestK = k
    print("Took: {}".format((time.time() - currTime)))

    plt.plot(kX, kY)
    plt.xlabel("k")
    plt.ylabel("E_val")
    plt.show()

    if NN:
        return lowestK, lowest
    return lowestK, lowest, W, centers


def classifyRBF(w, centers, x, r):
    transform = [1] + [gaussianKernel(distance(center, x) / r) for center in centers]
    return np.sign(np.dot(w, transform))


def errorRBF(w, centers, r, dTrain):
    error = 0
    for x, y in dTrain:
        if classifyRBF(w, centers, x, r) != y:
            error += 1
    return error / len(dTrain)


def plotCurved(weights, centers, xRange, yRange):
    xlist = np.linspace(-xRange, xRange, 250)
    ylist = np.linspace(-yRange, yRange, 250)
    X, Y = np.meshgrid(xlist, ylist)
    Z = []
    for x_row, y_row in zip(X, Y):
        z_row = zip(x_row, y_row)
        Z.append(z_row)

    new_Z = []
    k = len(centers)
    r = 2 / (np.sqrt(k))
    for z_row in Z:
        row = [classifyRBF(weights, centers, test_point, r) for test_point in z_row]
        new_Z.append(row)
    plt.contour(X, Y, new_Z, levels=[0])


def p1(trainingD, testingD):
    bestK, lowestCVError = plotCV(trainingD)
    plotDecisionRegion(3, trainingD, 1.1, 1.1, False)

    print("bestK:", bestK)
    print("cross validation error:", lowestCVError)
    print("test error:", errorNN(bestK, trainingD, testingD))


def p2(trainingD, testingD):
    bestK, lowestCVError, W, centers = plotCV(trainingD, False, 50)
    print("k chosen", bestK)
    plotCurved(W, centers, 1.1, 1.1)
    x1 = [x[0] for x, y in trainingD if y == 1]
    x2 = [x[1] for x, y in trainingD if y == 1]
    plt.scatter(x1, x2, color="blue")
    x1 = [x[0] for x, y in trainingD if y == -1]
    x2 = [x[1] for x, y in trainingD if y == -1]
    plt.scatter(x1, x2, color="red")
    plt.show()

    print("bestK:", bestK)
    print("cross validation error:", lowestCVError)
    print("test error:", errorRBF(W, centers, 2 / np.sqrt(bestK), testingD))


if __name__ == "__main__":
    file = "ZipDigits.all"
    data = Dtrain_Dtest(file)
    trainingData = data[0]
    testingData = data[1]
    # p1(trainingData, testingData)

    p2(trainingData, testingData)

import numpy as np
from matplotlib import pyplot as plt
from eighthOrder import eighthOrderTransform
from handwriting import generateData, plot


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

        intensities.append([horIntensity, verIntensity, 1 if dig == 1 else -1])
    return intensities, (minH, maxH), (minV, maxV)


def normalize(d, H, V):
    minH, maxH, minV, maxV = H[0], H[1], V[0], V[1]

    HScale, HShift = 2.0 / (maxH - minH), (minH + maxH) / 2.0
    VScale, VShift = 2.0 / (maxV - minV), (minV + maxV) / 2.0

    res = []
    for point in d:
        x1 = (point[0] - HShift) * HScale
        x2 = (point[1] - VShift) * VScale
        res.append([x1, x2, point[2]])
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


def transformData8th(dataset):
    newData = []
    for d in dataset:
        eighth = eighthOrderTransform(d[0], d[1])
        newData.append(eighth + [d[2]])
    return newData


def linearRegression(points, regularization):
    Z = np.array([x[:-1] for x in points])
    Y = np.transpose(np.array([x[-1] for x in points]))
    dimensions = len(Z[0])

    ZT = np.transpose(Z)
    ZTZ = np.dot(ZT, Z)
    eq1 = ZTZ + np.identity(dimensions) * regularization
    eq1 = np.linalg.inv(eq1)
    eq2 = np.dot(ZT, Y)

    w = np.dot(eq1, eq2)

    return w


def error(points, reg):
    Z = np.array([x[:-1] for x in points])
    Y = np.array([x[-1] for x in points])
    dimensions = len(Z[0])

    ZT = np.transpose(Z)
    ZTZ = np.dot(ZT, Z)
    ZTZ_reg = ZTZ + reg * np.identity(dimensions)
    inverse = np.linalg.inv(ZTZ_reg)
    H = np.dot(Z, inverse)
    H = np.dot(H, ZT)
    Yhat = np.dot(H, Y)

    CV = 0
    IN = 0
    for n in range(len(points)):
        yn = Y[n]
        yHatn = Yhat[n]
        Hnn = H[n][n]
        currSum = ((yHatn - yn) / (1 - Hnn)) ** 2

        IN += (yHatn - yn) ** 2
        CV += currSum
    return CV / len(points), IN / len(points)


def plotCurved(weight):
    x1 = np.arange(-1.1, 1.1, 0.1)
    x2 = np.arange(-1.1, 1.1, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    transWeight = [eighthOrderTransform(x1, x2) for x1, x2 in zip(X1, X2)]

    Z = []
    for tW in transWeight:
        w_tW = zip(weight, tW)
        weightL = [w * tW_ for (w, tW_) in w_tW]
        sumL = sum(weightL)
        Z.append(sumL)

    plt.contour(X1, X2, Z, [0])


def crossValidation(points, minReg=0.01, maxReg=2):
    x = np.arange(minReg, maxReg, 0.01)
    yCV = []
    yTEST = []
    minCVError = 1
    minINError = 1
    bestReg = 0
    for i in range(len(x)):
        reg = x[i]
        g = linearRegression(points, reg)
        CVError, TESTError = error(points, reg)
        yCV.append(CVError)
        yTEST.append(TESTError)

        if CVError < minCVError:
            minCVError = CVError
            bestReg = x[i]

        minINError = min(TESTError, minINError)

    plt.plot(x, yCV, label="E CV")
    plt.plot(x, yTEST, label="E test")
    plt.legend()
    plt.show()

    return bestReg, minCVError, minINError


if __name__ == "__main__":
    file = "ZipDigits.all"
    data = Dtrain_Dtest(file)
    trainingData = data[0]
    testingData = data[1]
    trainingData8th = transformData8th(trainingData)
    testingData8th = transformData8th(testingData)

    # 1
    # g = linearRegression(trainingData8th, 2)
    # plotCurved(g)
    # plot(trainingData)
    # plotCurved(g)
    # plot(testingData)

    # cross validation
    bestReg, bestRegError, bestINError = crossValidation(trainingData8th, 0.1, 5)
    print("bestReg: {}, E_cv: {}, E_test: {}".format(bestReg, bestRegError, bestINError))
    g = linearRegression(testingData8th, bestReg)
    plotCurved(g)
    plot(testingData)
    plt.show()

    print("done")

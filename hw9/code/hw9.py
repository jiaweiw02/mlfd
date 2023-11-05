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
    # plt.axis([-1.1, 1.1, -1.1, 1.1])
    # plt.show()

    # Z = weight[0] + weight[1] * X1 + weight[2] * X2 + weight[3] * \
    #     (X1 ** 2) + weight[4] * (X2 ** 2) + weight[5] * X1 * X2 + \
    #     weight[6] * (X1 ** 3) + weight[7] * (X2 ** 3) + weight[8] * \
    #     X1 * (X2 ** 2) + weight[9] * X2 * (X1 ** 2)
    #
    # plt.contour(X1, X2, Z, [0])
    # plt.axis([0, 140, 0, 120])


if __name__ == "__main__":
    file = "ZipDigits.all"
    data = Dtrain_Dtest(file)
    trainingData = data[0]
    testingData = data[1]

    # 1

    trainingData8th = transformData8th(trainingData)
    g = linearRegression(trainingData8th, 2)
    plotCurved(g)
    plot(trainingData)
    plotCurved(g)
    plot(testingData)

    print("done")

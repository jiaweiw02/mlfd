import time

import numpy as np
from knn import knn


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


def crossValidationNN(k, dTrain, dVal):
    err = 0
    for p in dVal:
        if knn(k, dTrain, p[0]) != p[1]:
            err += 1
    return err / len(dVal)


if __name__ == "__main__":
    file = "ZipDigits.all"
    data = Dtrain_Dtest(file)
    trainingData = data[0]
    testingData = data[1]

    currTime = time.time()
    lowest = 1
    lowestK = 0
    for k in range(1, len(trainingData) // 2):
        curr = crossValidationNN(k, trainingData, testingData)
        if curr < lowest:
            lowest = curr
            lowestK = k

    print("Took: {}".format((time.time() - currTime) * 100))

import numpy as np
from matplotlib import pyplot as plt


def distanceTwoPoints(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def plotWithWeight(w, x1, x2, l):
    a = -(w[1] / w[2])
    b = -(w[0] / w[2])
    start = [x1, x2]
    end = [a * x1 + b, a * x2 + b]
    plt.plot(start, end, label=l)


def validPoint(center, p, rad, sep):
    d = distanceTwoPoints(center, p)
    return True if rad < d < rad + sep else False


def generatePoints(center, rad, sep, xRange, yRange):
    while True:
        x = np.random.uniform(xRange[0], xRange[1])
        y = np.random.uniform(yRange[0], yRange[1])
        if validPoint(center, (x, y), rad, sep):
            return x, y


def createCircle(thk, rad, sep, points):
    centerTop = 0, 0
    centerBottom = rad + thk / 2, -1 * sep

    topXRange = (-1 * (rad + thk), (rad + thk))
    topYRange = (0, (rad + thk))
    bottomXRange = (-1 * (thk / 2), thk * 3 / 2 + 2 * rad)
    bottomYRange = (-1 * (sep + rad + thk), -1 * sep)

    plus1 = []
    minus1 = []

    for i in range(points):
        classification = np.random.randint(0, 2)
        if classification == 1:
            x, y = generatePoints(centerTop, rad, sep, topXRange, topYRange)
            plus1.append([x, y, 1])
        else:
            x, y = generatePoints(centerBottom, rad, sep, bottomXRange, bottomYRange)
            minus1.append([x, y, -1])

    # topX, topY = generatePoints(centerTop, rad, sep, points, topXRange, topYRange)
    # bottomX, bottomY = generatePoints(centerBottom, rad, sep, points, bottomXRange, bottomYRange)

    return plus1, minus1


def perceptron(points):
    w = [0, 0, 0]

    iter = 0
    while True:
        iter += 1
        hasMisclassified = False
        for p in points:
            x = [1, p[0], p[1]]
            output = p[2]

            dot_product = np.dot(w, x)
            if np.sign(dot_product) != output:
                for i in range(3):
                    w[i] += output * x[i]
                hasMisclassified = True
                break
        if not hasMisclassified:
            plotWithWeight(w, -15, 30, "PLA")
            return iter


def linearRegression(points):
    X = np.array([[1, x[0], x[1]] for x in points])
    y = np.array([x[2] for x in points])

    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    XTX_inverse = np.linalg.inv(XTX)
    psuedo = np.dot(XTX_inverse, XT)

    w = np.dot(psuedo, y)
    plotWithWeight(w, -15, 30, "LINEAR R")


def plotPoints(plus1, minus1):
    xTop = []
    yTop = []
    xBot = []
    yBot = []
    for p in plus1:
        xTop.append(p[0])
        yTop.append(p[1])

    for p in minus1:
        xBot.append(p[0])
        yBot.append(p[1])

    plt.plot(xTop, yTop, '.', color="blue", label="+1")
    plt.plot(xBot, yBot, '.', color="red", label="-1")
    plt.legend()
    plt.show()


def main():
    plus1, minus1 = createCircle(5, 10, 5, 2000)
    perceptron(plus1 + minus1)
    linearRegression(plus1 + minus1)
    plotPoints(plus1, minus1)

def main3_2(sep):
    plus1, minus1 = createCircle(5, 10, sep, 2000)

    return perceptron(plus1 + minus1)


main()

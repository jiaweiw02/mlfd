import numpy as np
from matplotlib import pyplot as plt


def p6_1():
    d = [
        ([1, 0], -1),
        ([0, 1], -1),
        ([0, -1], -1),
        ([-1, 0], -1),
        ([0, 2], 1),
        ([0, -2], 1),
        ([-2, 0], 1)
    ]

    # decisionRegion(3, dataset)
    # plotDecisionRegion(1, d, 3, 3, False)
    # plotDecisionRegion(3, d, 3, 3, False)
    plotDecisionRegion(1, d, 3, 3, True)
    plotDecisionRegion(3, d, 3, 3, True)


def knn(k, data, point):
    distances = [(p, y, distance(p, point)) for p, y in data]
    distances = sorted(distances, key=lambda x: x[2])
    distances = distances[:k]
    result = [y for point, y, dist in distances]

    return np.sign(sum(result))


def distance(p1, p2):
    x = (p1[0] - p2[0]) ** 2
    y = (p1[1] - p2[1]) ** 2
    return np.sqrt(x + y)


def plotDecisionRegion(k, data, xRange, yRange, transformed):
    x1s = np.arange(-xRange, xRange, 0.1)
    x2s = np.arange(-yRange, yRange, 0.1)
    transformedData = nonLinearTransform(data) if transformed else None

    for x1 in x1s:
        for x2 in x2s:
            testData = transformedData if transformed else data
            testPoint = transformPoint(x1, x2) if transformed else [x1, x2]
            Color = (0.68, 0.85, 0.90) if knn(k, testData, testPoint) == 1 else (1.0, 0.71, 0.76)
            plt.plot(x1, x2, marker="o", color=Color)

    for p, y in data:
        Color = "blue" if y == 1 else "red"
        Marker = "o" if y == 1 else "x"
        plt.plot(p[0], p[1], marker=Marker, color=Color)

    plt.show()


def nonLinearTransform(data):
    transformedData = []
    for x, y in data:
        [z1, z2] = transformPoint(x[0], x[1])
        transformedData.append(([z1, z2], y))
    return transformedData

def transformPoint(x1, x2):
    z1 = np.sqrt(x1 ** 2 + x2 ** 2)
    z2 = np.arctan(x2 / (x1 if x1 != 0 else 0.00000001))
    return [z1, z2]



if __name__ == "__main__":
    p6_1()

    # nonlinear transform

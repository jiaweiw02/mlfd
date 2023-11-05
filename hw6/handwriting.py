import numpy as np
from matplotlib import pyplot as plt
from p3_1 import linearRegression, plotWithWeight


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


def create_grayscale_image(data):
    # Create a 16x16 grid
    grid_size = 16
    grid = np.zeros((grid_size, grid_size))

    # Map the data to grayscale values
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            value = data[index]
            grid[i, j] = value

    # Display the grayscale image
    plt.imshow(grid, cmap='gray')
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()


def computeIntensity(data):
    intensities = []
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

        intensities.append([horIntensity, verIntensity, 1 if dig == 1 else -1])
    return intensities


def plot(intensities):
    oneX, oneY, fiveX, fiveY = [], [], [], []
    for int in intensities:
        if int[2] == 1:
            oneX.append(int[0])
            oneY.append(int[1])
        else:
            fiveX.append(int[0])
            fiveY.append(int[1])

    plt.plot(oneX, oneY, "x", label="1", color="red")
    plt.plot(fiveX, fiveY, "o", label="5", color="blue")
    plt.legend()
    plt.xlabel("horizontal intensity")
    plt.ylabel("vertical intensity")
    plt.show()


# takes in weight vector w1, w2, w3
# points in [x1,x2,y] format
def error(w, points):
    # calculate E_IN
    E = 0
    for p in points:
        x = [1] + p[:-1]
        output = p[-1]
        if np.sign(np.dot(w, x)) != output:
            E += 1
        # dotProduct = np.dot(w, x)
        # if np.sign(dotProduct) < 0:
        #     E += 1
    return E / len(points)
    # return E


def pocket(points, w, iterations):
    best_w = np.copy(w)
    best_accuracy = error(best_w, points)

    for i in range(iterations):
        # run PLA one update
        new_w = np.copy(best_w)

        index = np.random.randint(0, len(points))
        p = points[index]
        x = [1] + p[:-1]
        output = p[-1]
        while np.sign(np.dot(new_w, x)) == output:
            index = np.random.randint(0, len(points))
            p = points[index]
            x = [1] + p[:-1]
            output = p[-1]

        # found the misclassified point, update it
        for k in range(len(p) - 1):
            new_w[k] += output * x[k]

        accuracy = error(new_w, points)

        if accuracy < best_accuracy:
            print("updated", best_w, new_w, accuracy, best_accuracy)
            best_w = new_w
            best_accuracy = accuracy

    return best_w

def plotCurved(weight):
    x1 = np.arange(-1.1, 1.1, 0.1)
    x2 = np.arange(-1.1, 1.1, 0.1)
    X1, X2 = np.meshgrid(x1, x2)

    Z = weight[0] + weight[1] * X1 + weight[2] * X2 + weight[3] * \
        (X1 ** 2) + weight[4] * (X2 ** 2) + weight[5] * X1 * X2 + \
        weight[6] * (X1 ** 3) + weight[7] * (X2 ** 3) + weight[8] * \
        X1 * (X2 ** 2) + weight[9] * X2 * (X1 ** 2)

    plt.contour(X1, X2, Z, [0])
    plt.axis([0, 140, 0, 120])


# Assignment 7
# takes in points [x1,x2,...,xn,y] as a list
def linearReg_pocket(intensity, iterations, thirdOrder):
    w = linearRegression(intensity)
    pocketW = pocket(intensity, w, iterations)
    # print("pocketW = {:.4f}".format(error(pocketW, intensity)))
    if thirdOrder:
        plotCurved(pocketW)
    else:
        plotWithWeight(pocketW, 0, 75, "pocket")
    return w


def sigmoid(x):
    res = 1 / (1 + np.power(np.e, -x))
    # print(res)
    return res


def logisticReg(intensity, iterations, thirdOrder):
    learningRate = 0.1
    X = np.array([[1] + x[:-1] for x in intensity])
    y = np.array([x[-1] for x in intensity])
    w = np.array([1] + [np.random.uniform(-1, 1) for _ in range(len(intensity[0]) - 1)])

    for i in range(iterations):
        # calculate summation
        summation = 0
        N = len(intensity) - 1
        for j in range(N):
            summation += np.dot(y[j], X[j]) * sigmoid(y[j] * np.dot(w, X[j]))
        gradient = -(1 / N) * summation
        v = -gradient
        w += learningRate * v
    print("logisticRegW: {:.4f}".format(error(w, intensity)))

    if thirdOrder:
        plotCurved(w)
    else:
        plotWithWeight(w, 0, 75, "Logistic Reg")
    # plt.show()
    return w

    # sum = 0
    # for j in range(len(intensity)):
    #     numerator = np.dot(X[i], y[i])
    #     WT = np.transpose(w)
    #     denominator = 1 + np.e ** (np.dot())


def thirdOrderTransform(x):
    x1 = x[0]
    x2 = x[1]
    y = x[2]
    return np.array([1, x1, x2, x1 ** 2, x2 ** 2,
                     x1 * x2, x1 ** 3, x2 ** 3, x1 ** 2 * x2,
                     x1 * x2 ** 2, y])


def thirdOrderTransformPoints(points):
    res = []
    for p in points:
        res.append(thirdOrderTransform(p))
    return np.array(res)


if "__main__" == __name__:
    thirdOrder = True

    # data
    trainFile = "ZipDigits.all"
    trainData = generateData(trainFile)
    trainIntensity = computeIntensity(trainData)
    trainRealIntensity = thirdOrderTransformPoints(trainIntensity) if thirdOrder else trainIntensity

    # data
    testFile = "ZipDigits.test"
    testData = generateData(testFile)
    testIntensity = computeIntensity(testData)
    testRealIntensity = thirdOrderTransformPoints(testIntensity) if thirdOrder else testIntensity

    # linear regression w/ pocket
    print("begin linear reg w/ pocket")
    w1 = linearReg_pocket(trainRealIntensity, 1000, thirdOrder)
    plot(trainIntensity)
    plt.show()
    print("training: points: {}, error: {:.4f}".format(len(trainRealIntensity), error(w1, trainRealIntensity)))

    if thirdOrder:
        plotCurved(w1)
    else:
        plotWithWeight(w1, 0, 75, "pocket")
    plot(testIntensity)
    plt.show()
    print("test: points: {}, error: {:.4f}".format(len(testRealIntensity), error(w1, testRealIntensity)))

    # logistic reg
    print("\nbegin logistic reg gradient descent")
    w2 = logisticReg(trainIntensity, 1000, thirdOrder)
    plot(trainIntensity)
    plt.show()
    print("training: points: {}, error: {:.4f}".format(len(trainRealIntensity), error(w2, trainRealIntensity)))

    if thirdOrder:
        plotCurved(w2)
    else:
        plotWithWeight(w2, 0, 75, "Logistic Reg")
    plot(testIntensity)
    plt.show()
    print("test: points: {}, error: {:.4f}".format(len(testRealIntensity), error(w2, testRealIntensity)))




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

        intensities.append((horIntensity, verIntensity, 1 if dig == 1 else -1))
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
        x = [1, p[0], p[1]]
        output = p[2]
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
        x = [1, p[0], p[1]]
        output = p[2]
        while np.sign(np.dot(new_w, x)) == output:
            index = np.random.randint(0, len(points))
            p = points[index]
            x = [1, p[0], p[1]]
            output = p[2]

        # found the misclassified point, update it
        for k in range(3):
            new_w[k] += output * x[k]

        accuracy = error(new_w, points)

        if accuracy > best_accuracy:
            print("updated", best_w, new_w, accuracy, best_accuracy)
            best_w = new_w
            best_accuracy = accuracy

    return best_w


# Assignment 7
# takes in points [x1,x2,y] as a list, points are in tuple
def linearReg_pocket(intensity):
    w = linearRegression(intensity)
    pocket(intensity, w, 100)
    pocketW = pocket(intensity, w, 100)
    plotWithWeight(pocketW, 25, 75, "pocket")
    plot(intensity)


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def logisticReg(intensity, iterations):
    learningRate = 0.1
    X = np.array([[1, x[0], x[1]] for x in intensity])
    y = np.array([x[2] for x in intensity])
    w = np.array([np.random.uniform(-1, 1) for _ in range(3)])

    for i in range(iterations):
        # h = np.dot(X, w)
        # Error = sigmoid(h) - y
        # gradient = np.dot(np.transpose(X), Error) / len(intensity)
        # gradient = -1 * (np.dot(np.transpose(X), Error) / len(intensity))
        # v = -gradient
        # w -= learningRate * gradient
    plotWithWeight(w, 25, 75, "logistic reg")
    return w

    # sum = 0
    # for j in range(len(intensity)):
    #     numerator = np.dot(X[i], y[i])
    #     WT = np.transpose(w)
    #     denominator = 1 + np.e ** (np.dot())


# 1 is 1, -1 is 5

if "__main__" == __name__:
    trainFile = "ZipDigits.train"
    trainData = generateData(trainFile)
    trainIntensity = computeIntensity(trainData)
    linearReg_pocket(trainIntensity)

    print(logisticReg(trainIntensity, 1))
    plot(trainIntensity)
    plt.show()

    # testFile = "ZipDigits.test"
    # testData = generateData(testFile)
    # testIntensity = computeIntensity(testData)
    # plot(testIntensity)

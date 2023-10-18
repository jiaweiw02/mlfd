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

        intensities.append((horIntensity, verIntensity, dig))
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


def calcEIN():
    pass


#notes: just write different functions to calculate everything
# i don't even know what is wrong
def pocket(points, w, iterations):
    pocketW = w
    pointIndex = 0
    for i in range(iterations):
        currentW = np.copy(pocketW)
        # run PLA one update
        hasMisclassified = False
        for j in range(pointIndex, len(points)):
            p = points[j]
            x = [1, p[0], p[1]]
            output = -1 if p[2] == 1 else 1
            dot_product = np.dot(pocketW, x)
            if np.sign(dot_product) != output:
                pointIndex = j + 1
                hasMisclassified = True
                for k in range(3):
                    currentW[k] += output * x[k]
                break

        if not hasMisclassified:
            pocketW = currentW
            break

        # calculate E_IN
        E_INCurrent = 0
        E_INPocket = 0
        for p in points:
            x = [1, p[0], p[1]]
            output = -1 if p[2] == 1 else 1
            currentDot = np.dot(currentW, x)
            pocketDot = np.dot(pocketW, x)

            if np.sign(currentDot) != output:
                E_INCurrent += 1
            elif np.sign(pocketDot) != output:
                E_INPocket += 1

        print(E_INPocket, E_INCurrent)

        if E_INCurrent < E_INPocket:
            print("Pocket had {}, current had {}, updated {} to {}".format(E_INPocket, E_INCurrent, pocketW, currentW))
            pocketW = currentW

    return pocketW


# Assignment 7
# takes in points [x1,x2,y] as a list, points are in tuple
def linearReg_pocket(intensity):
    w = linearRegression(intensity)
    pocketW = pocket(intensity, w, 100)
    plotWithWeight(pocketW, 25, 75, "hi")
    plot(intensity)

# 1 is -1, 5 is 1

if "__main__" == __name__:
    trainFile = "ZipDigits.train"
    trainData = generateData(trainFile)
    trainIntensity = computeIntensity(trainData)
    linearReg_pocket(trainIntensity)

    plt.show()
    # plot(trainIntensity)

    # testFile = "ZipDigits.test"
    # testData = generateData(testFile)
    # testIntensity = computeIntensity(testData)
    # plot(testIntensity)

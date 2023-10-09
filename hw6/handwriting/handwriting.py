import numpy as np
from matplotlib import pyplot as plt


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

        intensities.append((dig, horIntensity, verIntensity))
    return intensities


def plot(intensities):
    oneX, oneY, fiveX, fiveY = [], [], [], []
    for int in intensities:
        if int[0] == 1:
            oneX.append(int[1])
            oneY.append(int[2])
        else:
            fiveX.append(int[1])
            fiveY.append(int[2])

    plt.plot(oneX, oneY, "x", label="1", color="red")
    plt.plot(fiveX, fiveY, "o", label="5", color="blue")
    plt.legend()
    plt.xlabel("horizontal intensity")
    plt.ylabel("vertical intensity")
    plt.show()


trainFile = "ZipDigits.train"
trainData = generateData(trainFile)
trainIntensity = computeIntensity(trainData)
plot(trainIntensity)

testFile = "ZipDigits.test"
testData = generateData(testFile)
testIntensity = computeIntensity(testData)
plot(testIntensity)





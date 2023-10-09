import numpy as np
from matplotlib import pyplot as plt


def readTrain():
    train = open("ZipDigits.train", 'r')
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


five_val = readTrain()[14][1:]
one_val = readTrain()[1][1:]
create_grayscale_image(five_val)
create_grayscale_image(one_val)

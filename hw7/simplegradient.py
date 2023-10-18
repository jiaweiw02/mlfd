import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x ** 2 + 2 * y ** 2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def gradientX(x, y):
    return 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def gradientY(x, y):
    return 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


pointsX = [0.1, 1, -0.5, -1]
pointsY = [0.1, 1, -0.5, -1]
minLoc = []
minValue = []

learningR = 0.1
iterations = 50
for j in range(4):
    x, y = pointsX[j], pointsY[j]
    xVal = []
    yVal = []
    currMinLoc = None
    currMinVal = np.inf
    for i in range(iterations):
        func = f(x, y)

        gX = gradientX(x, y)
        gY = gradientY(x, y)

        x -= learningR * gX
        y -= learningR * gY

        xVal.append(i)
        yVal.append(func)
        if func < currMinVal:
            currMinLoc = x, y
            currMinVal = func
    minLoc.append(currMinLoc)
    minValue.append(currMinVal)
    plt.plot(xVal, yVal, label="({},{})".format(pointsX[j], pointsY[j]))

print("x start & y start & x min & y min & min value\\\\")
for i in range(4):
    print("{} & {} & {} & {} & {}\\\\".format(pointsX[i], pointsY[i], minLoc[i][0], minLoc[i][1], minValue[i]))
print(minLoc, minValue)
plt.xlabel("iterations")
plt.ylabel("f(x, y)")
plt.legend()
plt.show()

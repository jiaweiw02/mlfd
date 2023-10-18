import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x ** 2 + 2 * y ** 2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def gradientX(x, y):
    return 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def gradientY(x, y):
    return 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


x = 0.1
y = 0.1
learningR = 0.1
iterations = 50

xVal = []
yVal = []

for i in range(iterations):
    func = f(x, y)

    gX = gradientX(x, y)
    gY = gradientY(x, y)

    x -= learningR * gX
    y -= learningR * gY

    xVal.append(i)
    yVal.append(func)

plt.plot(xVal, yVal)
plt.show()

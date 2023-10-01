import random
import numpy as np
from matplotlib import pyplot as plt


def algorithm(N):
    points = [random.uniform(-1, 1) for _ in range(2 * N)]

    a = 0
    b = 0

    for i in range(0, len(points), 2):
        p1, p2 = points[i], points[i + 1]
        a += (p1 + p2)
        b += (p1 * p2)

    a /= (2 * N)
    b /= -(2 * N)

    print("Average function g(x) = {:.4f}x + {:.4f}".format(a, b))

    eOut = 0
    for i in range(N):
        p1, p2 = random.uniform(-1, 1), random.uniform(-1, 1)
        currA = (p1 + p2)
        currB = -1 * (p1 * p2)
        x = random.uniform(-1, 1)
        g = currA * x + currB
        f = x**2
        E = (g - f)**2
        eOut += E
    eOut /= N
    print("Eout = {:.4f}".format(eOut))

    bias = 0
    for i in range(N):
        x = random.uniform(-1, 1)
        g = a * x + b
        f = x ** 2
        g_f = (g-f)**2
        bias += g_f
    bias /= N
    print("bias = {:.4f}".format(bias))

    var = 0
    for i in range(N):
        p1, p2 = random.uniform(-1, 1), random.uniform(-1, 1)
        currA = (p1 + p2)
        currB = -1 * (p1 * p2)
        x = random.uniform(-1, 1)
        g = currA * x + currB
        g_bar = a * x + b
        var += (g - g_bar)**2
    var /= N
    print("var = {:.4f}".format(var))

    print("bias + var = {:.4f}".format(bias + var))
    print("off actual value {:.4f} by {:.2f}%".format(eOut, 100 * abs(1 - ((bias + var)/eOut))))

    return a, b


def plot(xScale, yScale, a, b):
    x = np.linspace(-1 * xScale, xScale, 10)
    y = a * x + b

    targetX = np.linspace(-1 * yScale, yScale, 100)
    targetY = targetX ** 2

    plt.ylim(-1 * xScale, xScale)
    plt.ylim(-1 * yScale, yScale)
    plt.plot(x, y, label="g(x)")
    plt.plot(targetX, targetY, label="f(x)")
    plt.legend()
    plt.show()


def main():
    a, b = algorithm(1000)
    plot(1, 1, a, b)


main()

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    size = 20

    x = [0, 1]
    y = [-0.5, -0.5]

    plus1_x = []
    plus1_y = []
    minus1_x = []
    minus1_y = []

    for i in range(size):
        point_x = np.random.uniform(0, 1)
        point_y = np.random.uniform(-1, 0)

        if point_y >= -0.5:
            plus1_x.append(point_x)
            plus1_y.append(point_y)
        else:
            minus1_x.append(point_x)
            minus1_y.append(point_y)

    plt.plot(plus1_x, plus1_y, 'o')
    plt.plot(minus1_x, minus1_y, 'x')
    plt.plot(x, y)

    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.title('problem 1.4')

    plt.show()

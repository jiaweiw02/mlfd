import matplotlib.pyplot as plt
import numpy as np


def perceptron(points):
    w = [0, 0, 0]

    iter = 0
    while (True):
        hasMisclassified = False
        for p in points:
            x = [1, p[0], p[1]]
            output = p[2]

            dot_product = np.dot(w, x)
            if np.sign(dot_product) != output:
                for i in range(3):
                    w[i] += output * x[i]
                hasMisclassified = True
                break
        iter += 1
        if not hasMisclassified:
            a = -(w[1] / w[2])
            b = -(w[0] / w[2])
            start = [0, 1]
            end = [b, a + b]
            plt.plot(start, end, label="hypothesis")
            return iter


if __name__ == "__main__":

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    size = 1000

    # target function w = [0.5, 0, -1]
    x = [0, 1]
    y = [0.5, 0.5]

    points = []

    for i in range(size):
        point_x = np.random.uniform(0, 1)
        point_y = np.random.uniform(0, 1)

        if point_y >= 0.5:
            points.append([point_x, point_y, 1])
        else:
            points.append([point_x, point_y, -1])

    points_formatted = [[points[i][0] for i in range(len(points)) if points[i][2] == -1],
                        [points[i][1] for i in range(len(points)) if points[i][2] == -1],
                        [points[i][0] for i in range(len(points)) if points[i][2] == 1],
                        [points[i][1] for i in range(len(points)) if points[i][2] == 1]]

    plt.plot(points_formatted[0], points_formatted[1], 'x', label="-1")
    plt.plot(points_formatted[2], points_formatted[3], 'o', label="+1")
    plt.plot(x, y, label="target")

    print(perceptron(points))

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title('problem 1.4')

    plt.legend()
    plt.show()

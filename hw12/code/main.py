import numpy as np


def p1():
    m = 2
    initWeights = [[0.15 for _ in range(3)] for _ in range(2)]  # 3 x 2
    x = [2, 1]
    y = -1

    x1 = [1]
    # runs only two times
    for l in range(1, m + 1):
        s = np.dot(np.transpose(initWeights), x)
        x = np.tanh(s)
        print(x)
        initWeights = [0.15 for _ in range(m + 1)]


if "__main__" == __name__:
    p1()

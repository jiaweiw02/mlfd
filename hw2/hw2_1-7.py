import numpy
from matplotlib import pyplot as plt


def hoeffding(N, epsilon):
    e = 2.718281828
    return 2.0 * e ** (-2 * epsilon * epsilon * N)


plt.bar([0, 1 / 6, 2 / 6, 3 / 6, 1], [0.9023, 0.3897, 0.009766, 0, 0], width=0.05)

plt.plot([0, 1 / 6, 2 / 6, 3 / 6, 1],
         [2, hoeffding(6, 1 / 6), hoeffding(6, 2 / 6), hoeffding(6, 3 / 6), hoeffding(6, 1)])
plt.xlabel("epsilon")
plt.ylabel("probability")
plt.show()

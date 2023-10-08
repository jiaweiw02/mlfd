from p3_1 import main3_2
from matplotlib import pyplot as plt

iterations = []
ranges = [0.2 * i for i in range(1, 26)]
for i in ranges:
    i = round(i, 2)
    iter = main3_2(i)
    iterations.append(iter)


plt.plot(ranges, iterations)
plt.xlabel("sep values")
plt.ylabel("iterations")
plt.xlim(0.25, 5)
plt.ylim(0, 1000)
plt.show()

print(iterations)
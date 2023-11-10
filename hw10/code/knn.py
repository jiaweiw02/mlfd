import numpy as np
from matplotlib import pyplot as plt

def p6_1():
    d = [
        ([1, 0], -1),
        ([0, 1], -1),
        ([0, -1], -1),
        ([-1, 0], -1),
        ([0, 2], 1),
        ([0, -2], 1),
        ([-2, 0], 1)
    ]

def knn(k, data, point):
    distances = [distance(point, p) for p in data]


def distance(p1, p2):
    x = (p1[0] - p2[0]) ** 2
    y = (p1[1] - p2[1]) ** 2
    return np.sqrt(x + y)
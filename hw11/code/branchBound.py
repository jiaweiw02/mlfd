import time
import numpy as np
from matplotlib import pyplot as plt
from knn import distance


def generateData(points=10000, low=0, high=1):
    data = [[np.random.uniform(low, high), np.random.uniform(low, high)] for _ in range(points)]
    return data


def partition(k, data):
    centers = [data[0]]

    for i in range(k):
        furthestCenter = None
        furthestDist = 0
        for j in range(len(data)):
            pointX = data[j]
            dists = [distance(c, pointX) for c in centers]
            dists_pointToSet = min(dists)
            if dists_pointToSet > furthestDist:
                furthestCenter = data[j]
                furthestDist = dists_pointToSet
        centers.append(furthestCenter)
    return centers


def cluster(k, data):
    centers = partition(k, data)
    # all points go to their closest center
    centersPoints = {}
    for p in data:
        distances = [(x, distance(p, x)) for x in centers]
        minVal = min(distances, key=lambda x: x[1])
        centerX, centerDist = minVal
        centerX = tuple(centerX)
        if centerX not in centersPoints:
            centersPoints[centerX] = [p]
        else:
            centersPoints[centerX].append(p)

    # Varonoi
    centers = []
    for c in centersPoints.keys():
        pointsC = centersPoints[c]
        x1s = [x[0] for x in pointsC]
        x2s = [x[1] for x in pointsC]
        radii = [distance(p, list(c)) for p in pointsC]
        centers.append(([sum(x1s) / len(x1s), sum(x2s) / len(x2s)], max(radii), pointsC))
    return centers


# ^^^^ returns list of centers of ([x1, x2], radii, [points of that cluster])


def knnRuntime(k, data, point):
    distances = [(p, distance(p, point)) for p in data]
    distances = sorted(distances, key=lambda x: x[1])
    distances = distances[:k]
    return distances


# takes in point [x1, x2] NO y!!!!
def knnBranchBound(point, clusters):
    nearest = None
    nearestDist = np.inf
    for c in clusters:
        center = c[0]
        radius = c[1]
        clusterPoints = c[2]
        if nearest is not None:
            xToAny = distance(point, center) - radius
            if distance(point, nearest) <= xToAny:
                continue
        for p in clusterPoints:
            dist = distance(p, point)
            if dist < nearestDist:
                nearest = p
                nearestDist = dist

    return nearest


def gaussian(kCenter=10, kPoints=10000, scale=0.1):
    centers = generateData(kCenter)
    allP = {}
    pNormal = []
    for i in range(kPoints):
        currCenter = centers[np.random.randint(0, kCenter)]
        gaussianPoint = np.random.normal(loc=currCenter, scale=0.1)
        pNormal.append(gaussianPoint)
        currCenterTuple = tuple(currCenter)
        if currCenterTuple in allP:
            allP[currCenterTuple].append(gaussianPoint)
        else:
            allP[currCenterTuple] = [gaussianPoint]


    x1s = [x[0] for x in pNormal]
    x2s = [x[1] for x in pNormal]
    plt.scatter(x1s, x2s)
    plt.show()

    newCenters = []
    for c in allP.keys():
        pointsC = allP[c]
        radii = [distance(px, list(c)) for px in pointsC]
        newCenters.append((list(c), max(radii), pointsC))
    return newCenters, pNormal


if __name__ == "__main__":
    size = 10000
    # dataset = generateData(size)
    # clusters = cluster(10, dataset)
    # # 6.16 a
    # queryD = generateData(size)
    #
    # # knn time
    # startTime = time.time()
    # for p in queryD:
    #     knnRuntime(1, dataset, p)
    # print("knn brute force took: {:.2f}".format((time.time() - startTime)))
    #
    # # knn with branch and bound
    # startTime = time.time()
    # for queryPoint in queryD:
    #     knnBrandBound(queryPoint, clusters)
    # print("knn branch bound took: {:.2f}".format((time.time() - startTime)))

    # 6.16 b
    queryD = generateData(size)
    clusters, dataset = gaussian(10, size)
    # knn time
    startTime = time.time()
    for p in queryD:
        knnRuntime(1, dataset, p)
    print("knn brute force took: {:.2f}".format((time.time() - startTime)))

    # # knn with branch and bound
    startTime = time.time()
    for queryPoint in queryD:
        knnBranchBound(queryPoint, clusters)
    print("knn branch bound took: {:.2f}".format((time.time() - startTime)))


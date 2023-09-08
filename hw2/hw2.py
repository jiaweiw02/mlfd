import math
import matplotlib.pyplot as plt
import numpy as np
import time


def flipCoin():
    # list comprehension to generate coins
    # heads = 0, tails = 1
    coins = np.random.randint(0, 2, size=(1000, 10))

    # minHeads = 10
    # minIndex = 0
    # for i in range(len(coins)):
    #     headCount = coins[i].count(0)
    #     if headCount < minHeads:
    #         minIndex = i
    #         minHeads = headCount

    head_counts = np.sum(coins, axis=1)
    minIndex = np.argmin(head_counts)

    # print("head_counts: {}, minIndex: {}, coins[minIndex]: {}".format(head_counts, minIndex, coins[minIndex]))

    c1 = coins[0]
    c2 = coins[np.random.randint(0, 1000)]
    c3 = coins[minIndex]

    # mu = (c1.count(0) + c2.count(0) + c3.count(0)) / 30
    # return mu
    # return [c1.count(0) / 10, c2.count(0) / 10, c3.count(0) / 10]
    return [np.mean(c1), np.mean(c2), np.mean(c3)]


def generateData(experimentCount):
    startTime = time.time()

    results = [[]] * experimentCount
    for i in range(experimentCount):
        results[i] = flipCoin()

    print(results[0])
    print("finished {} experiments in {:.2f} seconds".format(experimentCount, time.time() - startTime))

    datasetV1 = {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0}
    datasetV2 = {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0}
    datasetV3 = {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0}

    for i in range(len(results)):
        datasetV1[results[i][0]] += 1
        datasetV2[results[i][1]] += 1
        datasetV3[results[i][2]] += 1

    return datasetV1, datasetV2, datasetV3


def plotData(dataset):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    keys = [dataset[0].keys(), dataset[1].keys(), dataset[2].keys()]
    values = [dataset[0].values(), dataset[1].values(), dataset[2].values()]

    ax[0].bar(keys[0], values[0], width=0.05)
    ax[0].set_ylabel('occurrence')
    ax[0].set_xlabel('probability of heads')
    ax[0].set_title('v1: first coin')

    ax[1].bar(keys[1], values[1], width=0.05)
    ax[1].set_ylabel('occurrence')
    ax[1].set_xlabel('probability of heads')
    ax[1].set_title('v2: random coin')

    ax[2].bar(keys[2], values[2], width=0.05)
    ax[2].set_ylabel('occurrence')
    ax[2].set_xlabel('probability of heads')
    ax[2].set_title('v3: least heads coin')
    plt.tight_layout()
    plt.show()


def hoeffding(N, epsilon):
    return 2.0 ** (-2 * epsilon * epsilon * N)


if "__main__" == __name__:
    experimentSize = 100000
    dataset = generateData(experimentSize)
    plotData(dataset)

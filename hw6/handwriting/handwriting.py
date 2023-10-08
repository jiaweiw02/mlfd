def readTrain():
    train = open("ZipDigits.train", 'r')
    allDigits = []
    for line in train:
        newData = [float(0) for i in range(257)]
        data = line.strip().split(" ")
        newData[0] = float(data[0])
        for i in range(1, len(data)):
            newData[i] = float(data[i])
        allDigits.append(newData)

    return allDigits


readTrain()
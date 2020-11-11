import numpy as np

xRowCount = 0
yRowCount = 0

mean = [10, 5]
cov = [[3,0], [0,10]]
x = np.random.multivariate_normal(mean, cov, 100)

for row in x:
    xRowCount += 1

xClassNums = []

for row in x:
    classNum = 1.0
    xClassNums.append([classNum])

xClassNums = np.array(xClassNums)
xAll = np.append(x, xClassNums, 1)

mean2 = [-10, -5]
cov2 = [[3,0], [0, 3]]
y = np.random.multivariate_normal(mean2, cov2, 100)

for row in y:
    yRowCount += 1

yClassNums = []

for row in y:
    classNum = -1.0
    yClassNums.append([classNum])

yClassNums = np.array(yClassNums)
yAll = np.append(y, yClassNums, 1)

allData = np.append(xAll, yAll, 0)
np.random.shuffle(allData)

splitData = np.array_split(allData, 2)

with open("./traindata.txt", "w") as file:
    np.savetxt(file, splitData[0])
    
with open("./testdata.txt", "w") as file:
    np.savetxt(file, splitData[1])
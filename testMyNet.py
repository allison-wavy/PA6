import numpy as np
from numpy import genfromtxt
import matplotlib as plot
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def normalize(self, dataArray):
       
        #sum all the columns
        plt.scatter(dataArray[:,0],dataArray[:,1], c="green", marker='o')
        colSums = np.sum(dataArray, 0)

        # get the mean and std from training
        with open("./mean_std.txt", "r") as file:
            arr = np.genfromtxt(file, dtype=float, delimiter=' ')
            means = [arr[0], arr[1]]
            stds = [arr[2], arr[3]]

        #create an empty numpy array with same shape as data array
        data = []
        #iterate through each row in the data array
        for item in dataArray:
            #for element in that row
            for index in range(2):
                #normalize that element
                x = (item[index] - means[0]) / stds[0]
                y = (item[index] - means[1]) / stds[1]
            #then create an array out of it
            arr = [x, y]
            #add that array to data numpy array
            data.append(arr)
        
        data = np.array(data)
        # print(data)

        return data

    def test(self, net, labels, fileName):
        correct = 0
        total = 0

        net.load_state_dict(torch.load('./myNet.pth'))
        #model = torch.load('./myNet.pth')

        #parse a numpy array from given data text file
        with open(fileName, "r") as file:
            npArr = np.genfromtxt(file, dtype=float, delimiter=' ')
        #print(npArr)

        #turn that numpy array into a data array excluding the class label in col 3
        npArrNoClass = np.delete(npArr, 2, 1)
        #print(npArrNoClass)

        #also turn that numpy array into a class label array excluding the data in col 1 and 2
        classLabelArray = np.delete(npArr, 0, 1)
        classLabelArray = np.delete(classLabelArray, 0, 1)
        #print(classLabelArray)

        #figure out how many rows and columns are in the data array for looping purposes
        numRows, numCols = npArrNoClass.shape
        #print(numRows)

        #normalize the data
        normalizedData = net.normalize(npArrNoClass)

        data = torch.from_numpy(normalizedData).type(torch.FloatTensor)

        #create an empty list for storing class predictions
        classPredictions = []

        #iterate through each row of the data
        for row in range(numRows):
            #get the output from the network
            out = net(data[row])

            #check if it's negative or positive
            if (out < 0):
                #if negative, give class -1.0
                prediction = -1.0
            elif (out >= 0):
                #if positive, give class 1.0
                prediction = 1.0
            #add that prediction to the predictions list
            classPredictions.append([prediction])
        #convert the completed predictions list into a numpy array
        classPredictions = np.array(classPredictions)
        #go through each element in the labels lists
        for index in range(len(classLabelArray)):
            #add 1 to the total
            total += 1
            #if the prediction matches the actual label given
            if classLabelArray[index] == classPredictions[index]:
                #add 1 to correct
                correct += 1
        
        print('Total: %d, Correct: %d' % (total, correct))
            
        print('Testing done. Accuracy: %d %%' % (100 * correct / total))

        plt.show()



def main():
    labels = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    net = Net()

    fileName = sys.argv[1]
    net.test(net, labels, fileName)


if __name__ == '__main__':
    main()
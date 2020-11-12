# Authors: Clint Lawson and Allison Cuba

# A linear classifier using Torch

import numpy as np
from numpy import genfromtxt
import matplotlib as plot
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x        

    def normalize(self, dataArray):
        #sum all the columns
        colSums = np.sum(dataArray, 0)
        #get the mean from the sum
        means = colSums / len(dataArray)
        #get the standard deviation of the columns
        stds = np.std(dataArray, 0)
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

        with open("./mean_std.txt", "w") as file:
            np.savetxt(file, means)
            np.savetxt(file, stds)

        return data

    def train(self, net, fileName):
        #set up number of epochs, loss function, and optimizer
        epochs = 25
        criterionMSE = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0) 

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
        #print(normalizedData)

        #convert those created arrays into tensors with torch
        classLabels = torch.from_numpy(classLabelArray).type(torch.FloatTensor)
        #print(classLabels)

        data = torch.from_numpy(normalizedData).type(torch.FloatTensor)
        #print(data)

        #loop through the whole training process 'epochs' times
        for e in range(epochs):
            #go through each row of the data tensor
            for row in range(numRows):
                optimizer.zero_grad() 
                #get the label for the specific row of data
                label = classLabels[row]
                #get the output of the network given the data of the specific row
                out = net(data[row])
                #compute the loss given the output and label
                loss = criterionMSE(out, label)
                # calculate the backward gradients for back propagation 
                loss.backward()
                #print('[Epoch %d] Loss: %.3f' % (e + 1, loss)) 

                # update parameters 
                optimizer.step()

        # found the two lines below at https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
        PATH = './myNet.pth'
        torch.save(net.state_dict(), PATH) 

        print('Training completed, network saved to myNet.pth')

def main():
    net = Net()
    fileName = sys.argv[1]
    net.train(net, fileName)

if __name__ == "__main__":
    main()
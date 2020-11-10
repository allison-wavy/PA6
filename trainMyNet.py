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
        # layer one has 2 inputs and 2 outputs
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc1(x))
        return x

    def normalize(self, fileName):
        data = []
        with open(fileName, "r") as file:
            for line in file:
                npArr = np.fromstring(line, dtype=float, sep=" ")
                data.append(npArr[0])
                data.append(npArr[1])
        print(data)
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        print(normalized)
        return normalized 

    def train(self, net, labels, fileName):
        epochs = 1
        for e in range(0, epochs):
            running_loss = 0.0
            # read a line from the file
            with open(fileName, "r") as file:
                for line in file:
                    # create a numpy array of floats from the line using space as separator
                    npArr = np.fromstring(line, dtype=float, sep=" ")
                    # print(npArr)
                    # get the class number from the 3rd column to compare against outputline
                    classNum = np.delete(npArr, 0, None)
                    classNum = np.delete(classNum, 0, None)
                    # remove the class number from the data, only use it to compare output
                    npArrNoClass = np.delete(npArr, 2, None)
                
                    # print(npArrNoClass)
                    # convert that data numpy array into a float tensor using torch
                    data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
                    # print(e, data)  
    
                    # measure mean squared loss
                    criterionMSE = nn.MSELoss()
                    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)
                    optimizer.zero_grad()      

                    out = net(data)
                    loss = criterionMSE(out, labels)
                    # calculate the backward gradients for back propagation 
                    loss.backward()
                    # update parameters 
                    optimizer.step()

                    # found the running_loss stuff in a pytorch example here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
                    running_loss += loss.item()
                    # print('[%d, %5d] loss: %.3f' % (epochs + 1, e + 1, running_loss))


                    # found the two lines below at https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
                    PATH = './myNet.pth'
                    torch.save(net.state_dict(), PATH)

def main():
    labels = torch.tensor([1.0, -1.0])
    net = Net()
    fileName = sys.argv[1]
    net.train(net, labels, fileName)
    normalized_data = net.normalize(fileName)
    #net.test(fileName, trainingSet)

if __name__ == "__main__":
    main()

# Questions I still have: 
    # 1. Are we falling forward anywhere?  Do we need to?
    # 2. In one of the lectures where he showed us how to define a NN, he created a network input tensor like this: x = torch.tensor([0.3, 0.7, -0.1])
        # then he declared labels in a separate tensor.  So do we need to declare an x? Or is that we're doing in lines 40-50?

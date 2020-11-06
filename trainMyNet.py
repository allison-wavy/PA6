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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc1(x))
        return x

    def train(self, net, labels, fileName):
        epochs = 1
        for e in range(0, epochs):
            # read a line from the file
            with open(fileName, "r") as file:
                for line in file:
                    # create a numpy array of floats from the line using space as separator
                    npArr = np.fromstring(line, dtype=float, sep=" ")
                    # get the class number from the 3rd column to compare against output
                    classNum = np.delete(npArr, 0, None)
                    classNum = np.delete(classNum, 0, None)
                    # remove the class number from the data, only use it to compare output
                    npArrNoClass = np.delete(npArr, 2, None)
                
                    # convert that data numpy array into a float tensor using torch
                    data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
                    print(e, data)  

                    criterionMSE = nn.MSELoss()
                    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)
                    optimizer.zero_grad()      

                    out = net(data)
                    print(out)
                    loss = criterionMSE(out, labels)
                    loss.backward()
                    print(loss)
                    optimizer.step()

                    #TODO: save the trained network to the myNet.pth file


def main():
    labels = torch.tensor([1.0, -1.0])
    net = Net()

    fileName = input("Please enter training data file name, including file extension: ")

    net.train(net, labels, fileName)
    #net.test(fileName, trainingSet)
          

if __name__ == "__main__":
    main()
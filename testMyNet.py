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
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

        data = []
        for i in range(0, len(normalized) / 2) :
            data.append([normalized[i], normalized[i + 1]])
            i += 2     

        return data 

    def test(self, net, labels, fileName):
        epochs = 1
        normalized_data = self.normalize(fileName)
        correct = 0
        total = 0

        epochs = 1
        model = torch.load('./myNet.pth')
        for e in range(0, epochs):
            # read a line from the file
            with open(fileName, "r") as file:
                for line in file:
                    total += 1
                    # create a numpy array of floats from the line using space as separator
                    npArr = np.fromstring(line, dtype=float, sep=" ")

                    # get the class number from the 3rd column to compare against outputline
                    classNum = np.delete(npArr, 0, None)
                    classNum = np.delete(classNum, 0, None)

                    # remove the class number from the data, only use it to compare output
                    # npArrNoClass = np.delete(npArr, 2, None)
                    npArrNoClass = np.array(normalized_data[e])
                    
                    # convert that data numpy array into a float tensor using torch
                    data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
                    # print(e, data)  


                    out = net(data)

                    _, predicted = torch.max(out.data, 0)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # print(out)
        print('Testing done.  Accuracy: %d %%' % (100 * correct / total))

def main():
    labels = torch.tensor([1.0, -1.9])
    net = Net()
    fileName = sys.argv[1]
    net.test(net, labels, fileName)
    

if __name__ == '__main__':
    main()
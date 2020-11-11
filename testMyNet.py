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
from trainMyNet import MEAN, STD

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # layer one has 2 inputs and 2 outputs
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

    def test(self, net, labels, fileName):
        epochs = 20
        correct = 0
        total = 0

        net.load_state_dict(torch.load('./myNet.pth'))

        with open(fileName, "r") as file:
            npArr = np.genfromtxt(file, dtype=float, delimiter=' ')
            npArrNoClass = np.delete(npArr, 2, 1)
            data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
            F.normalize(data, dim=0)
            #print(data)

        for e in range(0, epochs):
            total += 3
            out = net(data)
            print(out) #there should be epochs * 3 number of rows total among all tensors outputed
        
        #TODO: out and prediction are both the same every iteration right now
        for row in range(3):
            _, predicted = torch.max(out.data[row], 0)
            print(predicted)
            correct += (predicted == labels[row]).sum().item()


        #TODO: somehow need to include the mean and standard deviation as calculated via training
        #Also, need to change how correct things are found.
        
        print('Total: %d, Correct: %d' % (total, correct))
            
        print('Testing done. Accuracy: %d %%' % (100 * correct / total))

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
    labels = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    net = Net()

    fileName = sys.argv[1]
    net.test(net, labels, fileName)


if __name__ == '__main__':
    main()
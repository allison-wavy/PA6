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
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x        

    def train(self, net, labels, fileName):
        epochs = 20
        criterionMSE = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0) 

        with open(fileName, "r") as file:
            npArr = np.genfromtxt(file, dtype=float, delimiter=' ')
            npArrNoClass = np.delete(npArr, 2, 1)
            data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
            F.normalize(data, dim=0)

        for e in range(0, epochs):
            running_loss = 0.0
                    
            optimizer.zero_grad() 

            out = net(data)
            #print(out)

            loss = criterionMSE(out, labels)
            # calculate the backward gradients for back propagation 
            loss.backward()
            #print(loss)

            # found the running_loss stuff in a pytorch example here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
            running_loss += loss.item()
            #print('[Epoch %d] loss: %.3f' % (e + 1, running_loss))

            # update parameters 
            optimizer.step()

        #TODO: We still need to calculate the mean and standard deviation for use in testing
        #according to the assignment instructions

        # found the two lines below at https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
        PATH = './myNet.pth'
        torch.save(net.state_dict(), PATH) 

        print('Training completed, network saved to myNet.pth')

def main():
    labels = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    net = Net()
    fileName = sys.argv[1]
    net.train(net, labels, fileName)

if __name__ == "__main__":
    main()
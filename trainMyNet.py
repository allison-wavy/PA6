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

    def train(self, net, fileName):
        #set up number of epochs, loss function, and optimizer
        epochs = 20
        criterionMSE = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0) 

        #parse a numpy array from given data text file
        with open(fileName, "r") as file:
            npArr = np.genfromtxt(file, dtype=float, delimiter=' ')

        #turn that numpy array into a data array excluding the class label in col 3
        npArrNoClass = np.delete(npArr, 2, 1)
        #also turn that numpy array into a class label array excluding the data in col 1 and 2
        classLabelArray = np.delete(npArr, 0, 1)
        classLabelArray = np.delete(classLabelArray, 0, 1)
        #figure out how many rows and columns are in the data array for looping purposes
        numRows, numCols = npArrNoClass.shape
        #convert those created arrays into tensors with torch
        classLabels = torch.from_numpy(classLabelArray).type(torch.FloatTensor)
        data = torch.from_numpy(npArrNoClass).type(torch.FloatTensor)
        #normalize the data tensor
        F.normalize(data, dim=0) #TODO: substitute this for the mean/std version
        #for keeping track of the total loss
        #running_loss = 0.0 #TODO: Do we need running loss?        

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
                print(loss) 
                # found the running_loss stuff in a pytorch example here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
                #running_loss += loss.item() #TODO: Do we need running loss?
                #print('[Epoch %d] loss: %.3f' % (e + 1, running_loss))

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
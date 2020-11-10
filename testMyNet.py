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
        x = self.fc1(x)
        x = F.relu(self.fc1(x))
        return x

    def train(self, net, labels, fileName):
        epochs = 10
        for e in range(0, epochs):
            with open(fileName, "r") as file:
                for line in file:
                    npArr = np.fromstring(line, dtype=float, sep="")

                    classNum = np.delet(npArr, 0, None)

        correct = 0
        total = 0

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def main():
    fileName = sys.argv[1]
    net = Net()
    model = torch.load('./myNet.pth')


if __name__ == '__main__':
    main()
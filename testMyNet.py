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

    def test(self, fileName, trainingSet):
        # get info from myNet.pth
        # use that info with the info from fileName to calculate accuracy
        correct = 0
        total = 0

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def main():
    fileName = sys.argv[1]
    net = Net()
    model = torch.load('./myNet.pth')
    model.eval()

if __name__ == '__main__':
    main()
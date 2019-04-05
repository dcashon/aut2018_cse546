import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# D. Cashon
# 12 2 2018
# CSE546
# Homework 4
# Define the three different neural nets


# Net from tutorial ending
class DankNet(nn.Module):
    def __init__(self):
        super(DankNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# FirstNet Architecture
# Zero hidden layers, fully connected, linear
# Hyperparameters: none + (momentum, step_size) (2)
class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        # No convolutions
        # No pools
        self.fc1 = nn.Linear(32*32*3, 10, bias=True)
    def forward(self, x):
        # No pools
        # Modify x to vector
        x = x.view(-1, 32*32*3)
        # no relu needed
        x = self.fc1(x)
        return x

# SecondNet Architecture
# Fully connected, 1 hidden layer, relu+linear
# Hyperparameters: M + (momentum, step_size) (3)
class SecondNet(nn.Module):
    def __init__(self, M):
        super(SecondNet, self).__init__()
        self.M = M #hyperparameter
        # No convolutions
        # No pools
        # forward to hidden layer
        self.fc1 = nn.Linear(32*32*3, self.M, bias=True)
        # forward to output layer
        self.fc2 = nn.Linear(self.M, 10, bias=True)

    def forward(self, x):
        # No pools
        # Modify x to vector
        x = x.view(-1, 32*32*3)
        # Relu
        x = F.relu(self.fc1(x))
        # output
        x = self.fc2(x)
        return x

# ThirdNet Architecture
# Fully connected, 1 convolution and maxpool, 1 linear
# Hyperparameters: M, p, n + (momentum, step_size) (5)
class ThirdNet(nn.Module):
    def __init__(self, M, p, N):
        super(ThirdNet, self).__init__()
        self.M = int(M) # filter output channel number
        self.p = int(p) # convolutional filter size
        self.N = int(N) # maxpool size
        self.conv1 = nn.Conv2d(3, self.M, self.p)
        self.pool = nn.MaxPool2d(self.N)
        # forward to output layer
        self.fc1 = nn.Linear(int(((33-self.p)/self.N)**2 * self.M), 10, bias=True)

    def forward(self, x):
        # pass x to the convoltion layer and relu
        x = F.relu(self.conv1(x))
        #print(x.size()) #debug
        # SIZE(X) = (33-p)(33-p)M
        x = self.pool(x)
        #print(x.size()) #debug
        # SIZE(X) = ((33-p)/N)((33-p)/N)M
        #print(int(((33-self.p)/self.N)**2 * self.M))
        x = x.view(-1, int(((33-self.p)/self.N)**2 * self.M))
        # RESHAPE TO VEC
        # PASS TO FULLY CONNECTED LAYER
        x = self.fc1(x)
        return x


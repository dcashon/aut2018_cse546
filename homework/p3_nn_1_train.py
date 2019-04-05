import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from neural_nets import FirstNet, SecondNet, ThirdNet

# D. Cashon
# Code for training the neural nets and hyperparameter selection
# storage for errors, hyperparameter values
train_errors = []
test_errors = []
hyperparameters = []


# LOAD IN THE DATA
#############################################################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# EVALUATE AND TRAIN THE NET
#############################################################
net = FirstNet()
# examine net

# loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(6):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        # EVALUATE ON THE DATASET EACH EPOCH
        test_correct = 0
        test_total = 0
        train_correct = 0
        train_total = 0
    with torch.no_grad():
        # EVALUATE ON THE TRAINING SET
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        # EVALUATE ON THE TEST SET
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 60000 train images: %d %%' % (
        100 * train_correct / train_total))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * test_correct / test_total))
    train_errors.append(100 * train_correct / train_total)
    test_errors.append(100 * test_correct / test_total)

plt.scatter([x for x in range(len(train_errors))],train_errors, label='train_error')
plt.scatter([x for x in range(len(train_errors))], test_errors, label='test_error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
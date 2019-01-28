# -*- coding: utf-8 -*-

"""
Implementation of a CNN for for dogs and cats dataset.
"""

__authors__ = "Jimmy Leroux"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__studentid__ = "1024610"

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, padding=0),
            nn.ReLU())
            # 26 24 12 10 8 4 2
        self.denses = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            # nn.Dropout2d(p=0.20),
            nn.Linear(512, 10))

    def forward(self, x):
        output = x
        for conv in self.convlayers:
            output = conv(output)
        output = output.view(-1, 256 * 2 * 2)
        for dense in self.denses:
            output = dense(output)
        return output

    def train_(self, trainloader, validloader, device, num_epoch=10, lr=0.1):
            """
            Train function for the network.

            Inputs:
            -------
            cnn: conv_net instance we want to train.
            trainloader: pytorch Dataloader instance containing the
                            training data.
            testloader: pytorch Dataloader instance containing the test
                        data.
            num_epoch: number of training epoch.
            lr: Learning rate for the SGD.

            Returns:
            -------
            loss_train: normalized loss on the train data at after
                        each epoch.
            loss_valid: normalized loss on the test data at after each
                        epoch.
            err_train: total error on the train set after each epoch.
            err_valid: total error on the test set after each epoch.
            """

            criterion = nn.CrossEntropyLoss()

            loss_train = []
            loss_valid = []
            err_train = []
            err_valid = []
            for epoch in range(num_epoch):
                self.train()
                optimizer = optim.SGD(self.parameters(), lr=lr)
                for datas in trainloader:
                    inputs, labels = datas
                    optimizer.zero_grad()
                    outputs = self(inputs.to(device))
                    loss = criterion(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()

                running_loss_train = 0.0
                running_loss_valid = 0.0
                correct = 0.
                total = 0.
                self.eval()
                with torch.no_grad():
                    for datas in trainloader:
                        inputs, labels = datas
                        outputs = self(inputs.to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(device)).sum(
                            ).item()
                        loss = criterion(outputs, labels.to(device))
                        running_loss_train += loss.item() / len(train)
                err_train.append(1 - correct / total)

                correct = 0.
                total = 0.
                with torch.no_grad():
                    for datas in validloader:
                        inputs, labels = datas
                        outputs = self(inputs.to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(device)).sum(
                            ).item()
                        loss = criterion(outputs, labels.to(device))
                        running_loss_valid += loss.item() / len(valid)
                err_valid.append(1 - correct / total)

                loss_train.append(running_loss_train)
                loss_valid.append(running_loss_valid)
                print('Epoch: {}'.format(epoch))
                print('Train loss: {0:.4f} Train error: {1:.2f}'.format(
                    loss_train[epoch], err_train[epoch]))
                print('Test loss: {0:.4f} Test error: {1:.2f}'.format(
                   loss_valid[epoch], err_valid[epoch]))

            print('Finished Training')
            return (loss_train, loss_valid, err_train, err_valid)

# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class ConvNet(nn.Module):
    """
    Implementation of a convolutional neural network.
    """

    def __init__(self, dataset="mnist"):
        super(ConvNet, self).__init__()
        self.dataset = dataset
        # We define our model depending of the dataset.
        if self.dataset == "cat_and_dogs":
            # Define the convolutional layers as well as the activations.
            self.convlayers = nn.Sequential(
                nn.Conv2d(3, 64, 5, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 5, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 5, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 5, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512))
            # 60 56 28 24 20 10
            # Define the dense layers at the end of the network. These are
            # used to make the final predictions.
            self.denses = nn.Sequential(
                nn.Linear(512 * 4 * 4, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.20),
                nn.ReLU(),
                nn.Linear(1024, 2))
            # Note that there is no activation for the final layer.
            # this is because the softmax is embedded in the
            # criteron

        if self.dataset == "mnist":
            self.convlayers = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=0),
                nn.ReLU())
            # 26 24 12 10 8 4 2
            self.denses = nn.Sequential(
                nn.Linear(256 * 2 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 10))

    def forward(self, x):
        """
        Forward method for the network.

        Parameters:
        ----------
        x (torch.tensor): Input of the network. Can be in minibatches.

        Returns:
        --------
        output (torch.tensor): Output of the network, befor the softmax.
        """

        # We chose the right forward method depending of the chosen
        # dataset.
        if self.dataset == "cat_and_dogs":
            output = x
            for conv in self.convlayers:
                output = conv(output)
            output = output.view(-1, 512 * 4 * 4)
            for dense in self.denses:
                output = dense(output)
            return output

        if self.dataset == "mnist":
            output = x
            for conv in self.convlayers:
                output = conv(output)
            output = output.view(-1, 256 * 2 * 2)
            for dense in self.denses:
                output = dense(output)
            return output

    def train_(
            self, train, valid, device, num_epoch=10,
            lr=0.1, batchsize=256):
        """
        Train function for the network.

        Inputs:
        -------
        train (torch dataset): Torch dataset for the trainset.
        test (torch dataset): torch dataset for the testset.
        device (str): Name of the device to do the compute on.
        num_epoch (int): Number of epoch for the train.
        lr (float): Learning rate.
        batchsize (int): Size of the minibatches.

        Returns:
        -------
        loss_train (list): normalized loss on the train data at after
                            each epoch.
        loss_valid (list): normalized loss on the test data at after each
                            epoch.
        err_train (list): total error on the train set after each epoch.
        err_valid (list): total error on the test set after each epoch.
        """

        trainloader = torch.utils.data.DataLoader(
            train, batch_size=batchsize, shuffle=True)
        validloader = torch.utils.data.DataLoader(
            valid, batch_size=batchsize, shuffle=False)

        # Define the optimization criterion.
        criterion = nn.CrossEntropyLoss()

        loss_train = []
        loss_valid = []
        err_train = []
        err_valid = []
        for epoch in range(num_epoch):
            # Set the model in train mode.
            self.train()
            # Define the optimizer as SDG.
            if self.dataset == "cat_and_dogs":
                lrd = lr * (1 / (1 + 10*epoch/num_epoch))
                optimizer = optim.SGD(self.parameters(), lr=lrd)
            else:
                optimizer = optim.Adam(self.parameters(), lr=lr)
            for datas in trainloader:
                inputs, labels = datas
                optimizer.zero_grad()
                outputs = self(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

            running_loss_train = 0.0
            correct = 0.
            total = 0.
            # Set the model in evaluation mode.
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

                    running_loss_train += loss.item() * labels.size(0)
            err_train.append(1 - correct / total)
            loss_train.append(running_loss_train / total)

            running_loss_valid = 0.0
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
                    running_loss_valid += loss.item() * labels.size(0)
            err_valid.append(1 - correct / total)
            loss_valid.append(running_loss_valid / total)

            print('Epoch: {}'.format(epoch))
            print('Train loss: {0:.6f} Train error: {1:.4f}'.format(
                loss_train[epoch], err_train[epoch]))
            print('Test loss: {0:.6f} Test error: {1:.4f}'.format(
                loss_valid[epoch], err_valid[epoch]))

        print('Finished Training')
        return (loss_train, loss_valid, err_train, err_valid)

    def prediction(self, datas, batchsize, device):
        testloader = torch.utils.data.DataLoader(
            datas, batch_size=batchsize, shuffle=False)
        predictions = []
        classes = {
            0: "Cat",
            1: "Dog"
        }
        print(datas[0][0].shape)
        print(datas[0][0].data)
        
        #plt.imshow(datas[0][0].view(64,64,3)/255)
        #plt.show()
        self.eval()
        with torch.no_grad():
            for dat in testloader:
                inputs, _ = dat
                outputs = self(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
        with open('../../submission/submission.csv', mode='w') as submission:
            writer = csv.writer(submission, delimiter=',',
                quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(["id", "label"])
            for i in range(len(predictions)):
                predictions[i] = classes[predictions[i]]
                writer.writerow([str(i+1), predictions[i]])
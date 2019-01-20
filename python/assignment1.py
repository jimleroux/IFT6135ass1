# -*- coding: utf-8 -*-

"""
Kaggle competition for assignement 1 of IFT6135.
"""

__authors__ = "Jimmy Leroux"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__studentid__ = "1024610"

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop = nn.Dropout2d(p=0.05)
        self.dropfc = nn.Dropout(p=0.20)
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1) #was 5 96
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)  #44
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) #22
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)#9
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)        
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 31)

    def forward(self, x):
        x = self.drop((self.bn1(F.relu(self.conv1(x))))) #48
        x = self.drop(self.pool(self.bn2(F.relu(self.conv2(x))))) #22
        x = self.drop((self.bn3(F.relu(self.conv3(x)))))
        x = self.drop(self.pool(self.bn4(F.relu(self.conv4(x)))))
        x = self.drop((self.bn5(F.relu(self.conv5(x)))))
        x = self.drop(self.pool(self.bn6(F.relu(self.conv6(x)))))
        x = self.drop((self.bn7(F.relu(self.conv7(x)))))
        x = x.view(-1, 512 * 5 * 5)
        x = self.dropfc(self.bn8(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x

class kaggle_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        dat = self.data[index]
        if self.transforms is not None:
            dat = self.transforms(dat)
        return (dat,self.labels[index])
   
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
	pass

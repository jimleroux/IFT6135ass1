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
import os

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		
		self.convlayers = nn.Sequential(
			nn.Conv2d(3, 64, 5, padding=0),
			nn.ReLU(),
			#nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, 5, padding=0),
			nn.ReLU(),
			#nn.BatchNorm2d(64),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 128, 5, padding=0),
			nn.ReLU(),
			#nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, 5, padding=0),
			nn.ReLU(),
			#nn.BatchNorm2d(128),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(128, 256, 3, padding=0),
			nn.ReLU(),
			#nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(),
			#nn.BatchNorm2d(256),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU())
			#nn.BatchNorm2d(512),)
		# 60 56 28 24 20 10
		self.denses = nn.Sequential(
			nn.Linear(512 * 4 * 4, 1024),
			nn.ReLU(),
			#nn.BatchNorm1d(1024),
			#nn.Dropout2d(p=0.20),
			nn.Linear(1024, 2))
			
	def forward(self, x):
		output = x
		for conv in self.convlayers:
			output = conv(output)
		output = output.view(-1 ,512 * 4 * 4)
		for dense in self.denses:
			output = dense(output)
		return output 
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

	def train_(self, full_data, num_epoch=20, lr=0.1):
			"""
			Train function for the network.

			Inputs:
			-------
			cnn: conv_net instance we want to train.
			trainloader: pytorch Dataloader instance containing the training data.
			testloader: pytorch Dataloader instance containing the test data.
			num_epoch: number of training epoch.
			lr: Learning rate for the SGD.

			Returns:
			-------
			loss_train: normalized loss on the training data at after each epoch.
			loss_valid: normalized loss on the test data at after each epoch.
			err_train: total error on the training set after each epoch.
			err_valid: total error on the test set after each epoch.
			"""
			split = [
				int(0.8*len(full_data)),
				len(full_data)-int(0.8*len(full_data))
				]
			train, valid = torch.utils.data.dataset.random_split(full_data, split)
			trainloader = torch.utils.data.DataLoader(
				train, batch_size=64, shuffle=True)
			validloader = torch.utils.data.DataLoader(
				valid, batch_size=64, shuffle=True)

			criterion = nn.CrossEntropyLoss()
			
			loss_train = []
			loss_valid = []
			err_train = []
			err_valid = []
			for epoch in range(num_epoch):
				self.train()
				lrd = lr / (1 + 9*epoch/num_epoch)
				optimizer = optim.SGD(model.parameters(), lr=lr)		
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
				model.eval()
				with torch.no_grad():
					for datas in trainloader:
						inputs, labels = datas
						outputs = self(inputs.to(device))
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels.to(device)).sum().item()
						loss = criterion(outputs, labels.to(device))
						running_loss_train += loss.item() / len(train)
				err_train.append(1 - correct / total)
				
				correct = 0.
				total = 0.
				with torch.no_grad():
					for datas in validloader:
						inputs, labels = datas
						outputs = model(inputs.to(device))
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels.to(device)).sum().item()
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
			return loss_train, loss_valid, err_train, err_valid
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

CURRENT_DIR = os.getcwd()

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

class dataset(torch.utils.data.dataset.Dataset):
	def __init__(self, transforms=None):
		os.chdir("../dataset/trainset/")
		self.data = torchvision.datasets.ImageFolder(
			os.getcwd(), transform=transforms)
		os.chdir(CURRENT_DIR)

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

def train(cnn, full_data, num_epoch=20, lr=0.00001):
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
	split = [int(0.8*len(full_data)), len(full_data)-int(0.8*len(full_data))]
	train, valid = torch.utils.data.dataset.random_split(full_data, split)
	trainloader = torch.utils.data.DataLoader(
		train, batch_size=16, shuffle=True)
	validloader = torch.utils.data.DataLoader(
		valid, batch_size=16, shuffle=True)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(cnn.parameters(), lr=lr)
	
	loss_train = []
	loss_valid = []
	err_train = []
	err_valid = []
	for epoch in range(num_epoch):
		cnn.train()
		for data in trainloader:
			inputs, labels = data
			optimizer.zero_grad()
			outputs = cnn(inputs.to(device))
			loss = criterion(outputs, labels.to(device))
			loss.backward()
			optimizer.step()

		running_loss_train = 0.0
		running_loss_valid = 0.0
		correct = 0.
		total = 0.
		cnn.eval()
		with torch.no_grad():
			for data in trainloader:
				images, labels = data
				outputs = cnn(images.to(device))
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels.to(device)).sum().item()
				loss = criterion(outputs, labels.to(device))
				running_loss_train += loss.item()/len(train)
		err_train.append(1 - correct / total)
		
		correct = 0.
		total = 0.
		with torch.no_grad():
			for data in validloader:
				images, labels = data
				outputs = cnn(images.to(device))
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

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.cuda.manual_seed(10)
	plt.style.use('ggplot')
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.rc('axes', labelsize=15)
	
	# Define the transformations.
	transform = transforms.Compose(
		[transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	
	# Create, load and split the datas.
	full_data = dataset(transforms=transform)
	CNN = ConvNet().to(device)
	
	n_epoch = 10
	loss_train, loss_valid, e_train, e_valid = train(
		CNN, full_data, num_epoch=n_epoch)
	
	plt.figure()
	plt.plot(range(1,n_epoch+1),e_train, 'sk-', label='Train')
	plt.plot(range(1, n_epoch+1),e_valid, 'sr-', label='Valid')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(fontsize=25)
	plt.show()
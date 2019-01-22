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
import cnn

CURRENT_DIR = os.getcwd()

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
	CNN = cnn.ConvNet().to(device)
	
	n_epoch = 70
	loss_train, loss_valid, e_train, e_valid = CNN.train(
		full_data, num_epoch=n_epoch)
	
	plt.figure()
	plt.plot(range(1,n_epoch+1),e_train, 'sk-', label='Train')
	plt.plot(range(1, n_epoch+1),e_valid, 'sr-', label='Valid')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(fontsize=25)
	plt.show()
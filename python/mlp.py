# -*- coding: utf-8 -*-

"""
MLP
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



class NN(object):
    
    def __init__(
    		self,hidden_dims=(1024,2048),n_hidden=2,mode='train',
    		datapath=None,model_path=None):
    	pass

	def initialize_weights(self, n_hidden,dims):
		pass

	def forward(self, input,labels):
		pass

	def activation(self, input):
		pass

	def loss(self, prediction):
		pass

	def softmax(self, input):
		pass

	def backward(self, cache, labels):
		pass

	def update(self, grads):
		pass

	def train(self):
		pass

	def test(self):
		pass